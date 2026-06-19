use opencv::{
    core::{self, Point, Rect, Scalar, Vector},
    highgui, imgproc, prelude::*, videoio, Result as CvResult,
};
use std::net::UdpSocket;

const ARENA_LENGTH_CM: f64 = 142.0; 
const ARENA_WIDTH_CM: f64 = 77.0;   

const MIN_OBSTACLE_AREA: f64 = 20.0;
const MAX_OBSTACLE_AREA: f64 = 140.0;

// Радиус радара вокруг робота СМ
const DETECTION_RADIUS_CM: f64 = 20.0;

fn get_orientation_marker(
    frame: &core::Mat,
    lower_bound: core::Scalar,
    upper_bound: core::Scalar,
) -> CvResult<Option<Point>> {
    let mut hsv = core::Mat::default();
    imgproc::cvt_color_def(frame, &mut hsv, imgproc::COLOR_BGR2HSV)?;
    let mut mask = core::Mat::default();
    core::in_range(&hsv, &lower_bound, &upper_bound, &mut mask)?;
    
    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours(&mask, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

    let mut max_area = 0.0;
    let mut best_idx = -1;
    for i in 0..contours.len() {
        let area = imgproc::contour_area(&contours.get(i)?, false)?;
        if area > max_area && area > 10.0 {
            max_area = area;
            best_idx = i as i32;
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
    let robot_ip = "192.168.1.107:5000";
    println!("📡 Умный трекер запущен. Передача по UDP на {}", robot_ip);

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
    
    let mut frame = core::Mat::default();
    let mut black_mask = core::Mat::default();

    let lower_black = Scalar::new(0.0, 0.0, 0.0, 0.0);
    let upper_black = Scalar::new(180.0, 80.0, 60.0, 0.0); 

    let lower_blue = Scalar::new(100.0, 100.0, 50.0, 0.0);
    let upper_blue = Scalar::new(140.0, 255.0, 255.0, 0.0);
    let lower_pink = Scalar::new(155.0, 50.0, 50.0, 0.0);
    let upper_pink = Scalar::new(185.0, 255.0, 255.0, 0.0);

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

        // ИЩЕМ РОБОТА
        let front_marker = get_orientation_marker(&frame, lower_blue, upper_blue)?;
        let rear_marker = get_orientation_marker(&frame, lower_pink, upper_pink)?;
        let mut robot_packet = "RB:none".to_string();
        
        // Переменные для хранения координат робота
        let mut robot_cx_cm = 0.0;
        let mut robot_cy_cm = 0.0;
        let mut robot_found = false;

        if let (Some(front), Some(rear)) = (front_marker, rear_marker) {
            let center_x = (front.x + rear.x) / 2;
            let center_y = (front.y + rear.y) / 2;
            
            robot_cx_cm = center_x as f64 * px_to_cm_x;
            robot_cy_cm = center_y as f64 * px_to_cm_y;
            robot_found = true;
            robot_packet = format!("RB:{:.1},{:.1}", robot_cx_cm, robot_cy_cm);

            // Рисуем радиус обнаружения вокруг робота
            let radius_px = (DETECTION_RADIUS_CM / px_to_cm_x) as i32; // примерный перевод радиуса в пиксели
            imgproc::circle(&mut frame, Point::new(center_x, center_y), radius_px, Scalar::new(0.0, 255.0, 0.0, 0.0), 1, imgproc::LINE_8, 0)?;
            
            // Центр и вектор
            imgproc::circle(&mut frame, Point::new(center_x, center_y), 5, Scalar::new(0.0, 255.0, 0.0, 0.0), -1, imgproc::LINE_8, 0)?;
            imgproc::line(&mut frame, rear, front, Scalar::new(255.0, 255.0, 255.0, 0.0), 2, imgproc::LINE_8, 0)?;
        }

        // ИЩЕМ ПРЕПЯТСТВИЯ И ФИЛЬТРУЕМ ИХ ПО ДИСТАНЦИИ
        let mut contours = Vector::<Vector<Point>>::new();
        imgproc::find_contours(&mut black_mask, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;
        let mut obstacles_vector = Vec::new();

        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area_px = imgproc::contour_area(&contour, false)?;
            let area_cm2 = area_px * px2_to_cm2;
            let rect = imgproc::bounding_rect(&contour)?;

            let margin_x = 60; 
            let margin_y = 60; 
            if rect.x < margin_x || rect.y < margin_y || 
               rect.x + rect.width > frame_width as i32 - margin_x || 
               rect.y + rect.height > frame_height as i32 - margin_y {
                continue; 
            }

            if area_cm2 >= MIN_OBSTACLE_AREA && area_cm2 <= MAX_OBSTACLE_AREA {
                let obs_cx_cm = (rect.x + rect.width / 2) as f64 * px_to_cm_x;
                let obs_cy_cm = (rect.y + rect.height / 2) as f64 * px_to_cm_y;

                if robot_found {
                    // Считаем дистанцию от робота до кубика
                    let dx = obs_cx_cm - robot_cx_cm;
                    let dy = obs_cy_cm - robot_cy_cm;
                    let distance = (dx * dx + dy * dy).sqrt();

                    if distance <= DETECTION_RADIUS_CM {
                        // Кубик близко! Обводим красным и добавляем в пакет на отправку
                        obstacles_vector.push(format!("{:.1},{:.1}", obs_cx_cm, obs_cy_cm));
                        imgproc::rectangle(&mut frame, rect, Scalar::new(0.0, 0.0, 255.0, 0.0), 2, imgproc::LINE_8, 0)?;
                        imgproc::put_text(&mut frame, &format!("{:.1}cm", distance), Point::new(rect.x, rect.y - 5), imgproc::FONT_HERSHEY_SIMPLEX, 0.4, Scalar::new(0.0, 0.0, 255.0, 0.0), 1, imgproc::LINE_8, false)?;
                    } else {
                        // Кубик далеко. Обводим серым, игнорируем
                        imgproc::rectangle(&mut frame, rect, Scalar::new(128.0, 128.0, 128.0, 0.0), 1, imgproc::LINE_8, 0)?;
                    }
                } else {
                    // Если робот не найден, просто рисуем кубики серым, но никуда не отправляем
                    imgproc::rectangle(&mut frame, rect, Scalar::new(128.0, 128.0, 128.0, 0.0), 1, imgproc::LINE_8, 0)?;
                }
            }
        }

        // ОТПРАВКА ДАННЫХ
        let obstacles_packet = if obstacles_vector.is_empty() { "OB:none".to_string() } else { format!("OB:{}", obstacles_vector.join("|")) };
        let final_packet = format!("{};{}\n", robot_packet, obstacles_packet);
        
        // Отправка строго по UDP для максимальной скорости и минимального лага
        let _ = socket.send_to(final_packet.as_bytes(), robot_ip);

        imgproc::put_text(&mut frame, &final_packet.trim(), Point::new(10, 30), imgproc::FONT_HERSHEY_SIMPLEX, 0.5, Scalar::new(0.0, 255.0, 255.0, 0.0), 1, imgproc::LINE_8, false)?;
        
        let safe_zone = Rect::new(60, 60, frame_width as i32 - 120, frame_height as i32 - 120);
        imgproc::rectangle(&mut frame, safe_zone, Scalar::new(0.0, 100.0, 255.0, 0.0), 1, imgproc::LINE_8, 0)?;

        highgui::imshow("Brain Tracker", &frame)?;
        highgui::imshow("Debug: Black Mask", &black_mask)?;

        if highgui::wait_key(1)? == 113 { break; }
    }
    Ok(())
}