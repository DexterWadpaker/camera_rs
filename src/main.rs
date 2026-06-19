use opencv::{
    core::{self, Point, Rect, Scalar, Vector},
    highgui, imgproc, prelude::*, videoio, Result as CvResult,
};
use std::net::UdpSocket;

const ARENA_LENGTH_CM: f64 = 142.0; 
const ARENA_WIDTH_CM: f64 = 77.0;   

const MIN_OBSTACLE_AREA: f64 = 15.0; // Понизили, чтобы ловить кубики в низинах
const MAX_OBSTACLE_AREA: f64 = 140.0;

// Расширили лимиты для шасси робота (80 - низина, 700 - возвышенность)
const MIN_ROBOT_AREA: f64 = 80.0; 
const MAX_ROBOT_AREA: f64 = 700.0;

const DETECTION_RADIUS_CM: f64 = 20.0;

fn get_orientation_marker(
    frame: &core::Mat,
    lower_bound: core::Scalar,
    upper_bound: core::Scalar,
    window_name: &str,
) -> CvResult<Option<Point>> {
    let mut hsv = core::Mat::default();
    imgproc::cvt_color_def(frame, &mut hsv, imgproc::COLOR_BGR2HSV)?;
    let mut mask = core::Mat::default();
    core::in_range(&hsv, &lower_bound, &upper_bound, &mut mask)?;
    
    highgui::imshow(window_name, &mask)?;

    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours(&mask, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

    let mut max_area = 0.0;
    let mut best_idx = -1;
    for i in 0..contours.len() {
        let area = imgproc::contour_area(&contours.get(i)?, false)?;
        // Сильно снизили порог! Теперь метке достаточно быть площадью всего 3 пикселя
        if area > 3.0 && area < 500.0 {
            if area > max_area {
                max_area = area;
                best_idx = i as i32;
            }
        }
    }

    if best_idx >= 0 {
        let moments = imgproc::moments(&contours.get(best_idx as usize)?, false)?;
        if moments.m00 != 0.0 {
            return Ok(Some(Point::new((moments.m10 / moments.m00) as i32, (moments.m01 / moments.m00) as i32)));
        }
    }
    Ok(None)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let socket = UdpSocket::bind("0.0.0.0:8888").expect("Не удалось привязать сокет");
    // ТВОЙ IP-АДРЕС РОБОТА (Вставь актуальный)
    let robot_ip = "192.168.1.107:8888"; 
    println!("📡 Умный трекер запущен. Гибридный режим захвата включен.");

    let mut cap_opt = None;
    for index in 0..6 {
        if let Ok(try_cap) = videoio::VideoCapture::new(index, videoio::CAP_ANY) {
            if let Ok(opened) = try_cap.is_opened() {
                if opened {
                    println!("✅ КАМЕРА НА ИНДЕКСЕ: {}", index);
                    cap_opt = Some(try_cap);
                    break;
                }
            }
        }
    }

    let mut cap = match cap_opt {
        Some(c) => c,
        None => return Ok(()),
    };

    let frame_width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as f64;
    let frame_height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as f64;
    
    let px_to_cm_x = ARENA_LENGTH_CM / frame_width;
    let px_to_cm_y = ARENA_WIDTH_CM / frame_height;
    let px2_to_cm2 = px_to_cm_x * px_to_cm_y; 

    highgui::named_window("Brain Tracker", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("Debug: Black Mask", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("Debug: BLUE", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("Debug: PINK", highgui::WINDOW_AUTOSIZE)?;
    
    let mut frame = core::Mat::default();
    let mut black_mask = core::Mat::default();

    let lower_black = Scalar::new(0.0, 0.0, 0.0, 0.0);
    let upper_black = Scalar::new(180.0, 90.0, 60.0, 0.0); 

    // Максимально широкие фильтры для цветов (ловим даже грязные и темные оттенки в углах)
    let lower_blue = Scalar::new(85.0, 50.0, 40.0, 0.0);
    let upper_blue = Scalar::new(145.0, 255.0, 255.0, 0.0);
    
    let lower_pink = Scalar::new(140.0, 50.0, 40.0, 0.0);
    let upper_pink = Scalar::new(180.0, 255.0, 255.0, 0.0);

    loop {
        cap.read(&mut frame)?;
        if frame.empty() { continue; }

        let mut blurred = core::Mat::default();
        imgproc::gaussian_blur_def(&frame, &mut blurred, core::Size::new(7, 7), 0.0)?;

        let mut hsv = core::Mat::default();
        imgproc::cvt_color_def(&blurred, &mut hsv, imgproc::COLOR_BGR2HSV)?;
        core::in_range(&hsv, &lower_black, &upper_black, &mut black_mask)?;

        let kernel = core::Mat::default();
        let mut temp_mask = core::Mat::default();
        imgproc::erode(&black_mask, &mut temp_mask, &kernel, Point::new(-1, -1), 2, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;
        imgproc::dilate(&temp_mask, &mut black_mask, &kernel, Point::new(-1, -1), 4, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;

        let mut contours = Vector::<Vector<Point>>::new();
        imgproc::find_contours(&mut black_mask, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

        let mut robot_packet = "RB:none".to_string();
        let mut robot_found = false;
        let mut robot_cx_cm = 0.0;
        let mut robot_cy_cm = 0.0;
        
        // Переменные для Резервного трекинга
        let mut best_fallback_rect: Option<Rect> = None;
        let mut max_robot_area = 0.0;
        
        let mut raw_obstacles = Vec::new();

        // ШАГ 1: Поиск всех черных объектов (Препятствия + Шасси робота)
        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area_px = imgproc::contour_area(&contour, false)?;
            let area_cm2 = area_px * px2_to_cm2;
            let rect = imgproc::bounding_rect(&contour)?;

            // УМЕНЬШИЛИ МЕРТВУЮ ЗОНУ до 25 пикселей (теперь углы открыты!)
            let margin = 25; 
            if rect.x < margin || rect.y < margin || 
               rect.x + rect.width > frame_width as i32 - margin || 
               rect.y + rect.height > frame_height as i32 - margin {
                continue; 
            }

            if area_cm2 >= MIN_ROBOT_AREA && area_cm2 <= MAX_ROBOT_AREA {
                // Это кандидат на шасси робота (ищем самый большой подходящий кусок)
                if area_cm2 > max_robot_area {
                    max_robot_area = area_cm2;
                    best_fallback_rect = Some(rect);
                }
            } else if area_cm2 >= MIN_OBSTACLE_AREA && area_cm2 <= MAX_OBSTACLE_AREA {
                // Это кубик-препятствие
                let obs_cx_cm = (rect.x + rect.width / 2) as f64 * px_to_cm_x;
                let obs_cy_cm = (rect.y + rect.height / 2) as f64 * px_to_cm_y;
                raw_obstacles.push((rect, obs_cx_cm, obs_cy_cm));
            }
        }

        // ШАГ 2: Гибридный захват робота
        let front_marker = get_orientation_marker(&frame, lower_blue, upper_blue, "Debug: BLUE")?;
        let rear_marker = get_orientation_marker(&frame, lower_pink, upper_pink, "Debug: PINK")?;

        // Попытка А: Ищем по цветным меткам
        if let (Some(front), Some(rear)) = (front_marker, rear_marker) {
            let cx = (front.x + rear.x) / 2;
            let cy = (front.y + rear.y) / 2;
            robot_cx_cm = cx as f64 * px_to_cm_x;
            robot_cy_cm = cy as f64 * px_to_cm_y;
            robot_found = true;

            // Рисуем направление
            imgproc::line(&mut frame, rear, front, Scalar::new(255.0, 255.0, 255.0, 0.0), 2, imgproc::LINE_8, 0)?;
            imgproc::circle(&mut frame, front, 5, Scalar::new(255.0, 0.0, 0.0, 0.0), -1, imgproc::LINE_8, 0)?;
            
            // Если шасси совпало с метками, обводим зелёным
            if let Some(r_rect) = best_fallback_rect {
                imgproc::rectangle(&mut frame, r_rect, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0)?;
            }
        } 
        // Попытка Б: Резервный трекинг по шасси (если метки не видно)
        else if let Some(r_rect) = best_fallback_rect {
            let cx = r_rect.x + r_rect.width / 2;
            let cy = r_rect.y + r_rect.height / 2;
            robot_cx_cm = cx as f64 * px_to_cm_x;
            robot_cy_cm = cy as f64 * px_to_cm_y;
            robot_found = true;

            // Обводим желтым (сигнал, что едем в слепом режиме)
            imgproc::rectangle(&mut frame, r_rect, Scalar::new(0.0, 255.0, 255.0, 0.0), 2, imgproc::LINE_8, 0)?;
            imgproc::put_text(&mut frame, "TRACKING: CHASSIS ONLY", Point::new(r_rect.x, r_rect.y - 10), imgproc::FONT_HERSHEY_SIMPLEX, 0.5, Scalar::new(0.0, 255.0, 255.0, 0.0), 2, imgproc::LINE_8, false)?;
        }

        // Если робот найден (любым способом) - рисуем радар
        if robot_found {
            robot_packet = format!("RB:{:.1},{:.1}", robot_cx_cm, robot_cy_cm);
            
            let cx_px = (robot_cx_cm / px_to_cm_x) as i32;
            let cy_px = (robot_cy_cm / px_to_cm_y) as i32;
            let radius_px = (DETECTION_RADIUS_CM / px_to_cm_x) as i32; 
            
            imgproc::circle(&mut frame, Point::new(cx_px, cy_px), radius_px, Scalar::new(0.0, 255.0, 0.0, 0.0), 1, imgproc::LINE_8, 0)?;
        }

        // ШАГ 3: Радар препятствий (относительно робота)
        let mut obstacles_vector = Vec::new();
        for (rect, obs_cx_cm, obs_cy_cm) in raw_obstacles {
            if robot_found {
                let dx = obs_cx_cm - robot_cx_cm;
                let dy = obs_cy_cm - robot_cy_cm;
                let distance = (dx * dx + dy * dy).sqrt();

                if distance <= DETECTION_RADIUS_CM {
                    obstacles_vector.push(format!("{:.1},{:.1}", obs_cx_cm, obs_cy_cm));
                    imgproc::rectangle(&mut frame, rect, Scalar::new(0.0, 0.0, 255.0, 0.0), 2, imgproc::LINE_8, 0)?;
                    imgproc::put_text(&mut frame, &format!("{:.1}cm", distance), Point::new(rect.x, rect.y - 5), imgproc::FONT_HERSHEY_SIMPLEX, 0.4, Scalar::new(0.0, 0.0, 255.0, 0.0), 1, imgproc::LINE_8, false)?;
                } else {
                    imgproc::rectangle(&mut frame, rect, Scalar::new(128.0, 128.0, 128.0, 0.0), 1, imgproc::LINE_8, 0)?;
                }
            } else {
                imgproc::rectangle(&mut frame, rect, Scalar::new(128.0, 128.0, 128.0, 0.0), 1, imgproc::LINE_8, 0)?;
            }
        }

        // ШАГ 4: Отправка по UDP
        let obstacles_packet = if obstacles_vector.is_empty() { "OB:none".to_string() } else { format!("OB:{}", obstacles_vector.join("|")) };
        let final_packet = format!("{};{}\n", robot_packet, obstacles_packet);
        let _ = socket.send_to(final_packet.as_bytes(), robot_ip);

        imgproc::put_text(&mut frame, &final_packet.trim(), Point::new(10, 30), imgproc::FONT_HERSHEY_SIMPLEX, 0.5, Scalar::new(0.0, 255.0, 255.0, 0.0), 1, imgproc::LINE_8, false)?;
        
        // Отрисовка новых границ (теперь они шире)
        let safe_zone = Rect::new(25, 25, frame_width as i32 - 50, frame_height as i32 - 50);
        imgproc::rectangle(&mut frame, safe_zone, Scalar::new(0.0, 100.0, 255.0, 0.0), 1, imgproc::LINE_8, 0)?;

        highgui::imshow("Brain Tracker", &frame)?;
        highgui::imshow("Debug: Black Mask", &black_mask)?;

        if highgui::wait_key(1)? == 113 { break; }
    }
    Ok(())
}