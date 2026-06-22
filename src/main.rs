use opencv::{
    core::{self, Point, Rect, Scalar, Vector},
    highgui, imgproc, prelude::*, videoio, Result as CvResult,
};
use std::net::UdpSocket;

const ARENA_LENGTH_CM: f64 = 142.0; 
const ARENA_WIDTH_CM: f64 = 77.0;   

const MIN_OBSTACLE_AREA: f64 = 15.0; 
const MAX_OBSTACLE_AREA: f64 = 140.0;
const MIN_ROBOT_AREA: f64 = 80.0; 
const MAX_ROBOT_AREA: f64 = 700.0;

const DETECTION_RADIUS_CM: f64 = 20.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let socket = UdpSocket::bind("0.0.0.0:8888").expect("Не удалось привязать сокет");
    let robot_ip = "192.168.1.107:8888"; 
    println!("📡 Умный трекер запущен. Режим: Слияние Черного и Синего силуэтов.");

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
    highgui::named_window("Debug: Combined Mask", highgui::WINDOW_AUTOSIZE)?;
    
    let mut frame = core::Mat::default();

    // ФИЛЬТР ЧЕРНОГО (Гусеницы, тени под роботом, черные кубики)
    // Оставили V до 70, чтобы не цеплять зеленую траву
    let lower_black = Scalar::new(0.0, 0.0, 0.0, 0.0);
    let upper_black = Scalar::new(180.0, 255.0, 70.0, 0.0); 

    // ФИЛЬТР СИНИХ БЛИКОВ (Отражающий пластик корпуса)
    // H: 90-140 (Синий спектр)
    // S: 40-255 (Любая насыщенность)
    // V: 40-150 (Разрешаем ему быть гораздо светлее черного, чтобы поймать блик от окна)
    let lower_blue_glare = Scalar::new(90.0, 40.0, 40.0, 0.0);
    let upper_blue_glare = Scalar::new(140.0, 255.0, 150.0, 0.0);

    loop {
        cap.read(&mut frame)?;
        if frame.empty() { continue; }

        let mut blurred = core::Mat::default();
        imgproc::gaussian_blur_def(&frame, &mut blurred, core::Size::new(7, 7), 0.0)?;

        let mut hsv = core::Mat::default();
        imgproc::cvt_color_def(&blurred, &mut hsv, imgproc::COLOR_BGR2HSV)?;

        // Создаем две отдельные маски
        let mut black_mask = core::Mat::default();
        core::in_range(&hsv, &lower_black, &upper_black, &mut black_mask)?;

        let mut blue_mask = core::Mat::default();
        core::in_range(&hsv, &lower_blue_glare, &upper_blue_glare, &mut blue_mask)?;

        // СЛИЯНИЕ: Складываем черное и синее в одну общую картинку
        let mut combined_mask = core::Mat::default();
        core::bitwise_or(&black_mask, &blue_mask, &mut combined_mask, &core::Mat::default())?;

        // СКЛЕИВАЕМ СИЛУЭТ (Завариваем швы между синим пластиком и черными гусеницами)
        let kernel = core::Mat::default();
        let mut temp_mask = core::Mat::default();
        imgproc::dilate(&combined_mask, &mut temp_mask, &kernel, Point::new(-1, -1), 6, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;
        imgproc::erode(&temp_mask, &mut combined_mask, &kernel, Point::new(-1, -1), 2, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;

        let mut contours = Vector::<Vector<Point>>::new();
        // Ищем контуры уже по нашей ОБЪЕДИНЕННОЙ маске
        imgproc::find_contours(&mut combined_mask, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

        let mut robot_packet = "RB:none".to_string();
        let mut robot_found = false;
        let mut robot_cx_cm = 0.0;
        let mut robot_cy_cm = 0.0;
        
        let mut best_robot_rect: Option<Rect> = None;
        let mut max_robot_area = 0.0;
        let mut raw_obstacles = Vec::new();

        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area_px = imgproc::contour_area(&contour, false)?;
            let area_cm2 = area_px * px2_to_cm2;
            let rect = imgproc::bounding_rect(&contour)?;

            let margin = 45; 
            if rect.x < margin || rect.y < margin || 
               rect.x + rect.width > frame_width as i32 - margin || 
               rect.y + rect.height > frame_height as i32 - margin {
                continue; 
            }

            if area_cm2 >= MIN_ROBOT_AREA && area_cm2 <= MAX_ROBOT_AREA {
                if area_cm2 > max_robot_area {
                    max_robot_area = area_cm2;
                    best_robot_rect = Some(rect);
                }
            } else if area_cm2 >= MIN_OBSTACLE_AREA && area_cm2 <= MAX_OBSTACLE_AREA {
                let obs_cx_cm = (rect.x + rect.width / 2) as f64 * px_to_cm_x;
                let obs_cy_cm = (rect.y + rect.height / 2) as f64 * px_to_cm_y;
                raw_obstacles.push((rect, obs_cx_cm, obs_cy_cm));
            }
        }

        if let Some(r_rect) = best_robot_rect {
            robot_cx_cm = (r_rect.x + r_rect.width / 2) as f64 * px_to_cm_x;
            robot_cy_cm = (r_rect.y + r_rect.height / 2) as f64 * px_to_cm_y;
            robot_found = true;

            imgproc::rectangle(&mut frame, r_rect, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0)?;
            imgproc::put_text(&mut frame, "ROBOT (BLK+BLU)", Point::new(r_rect.x, r_rect.y - 10), imgproc::FONT_HERSHEY_SIMPLEX, 0.5, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, false)?;

            robot_packet = format!("RB:{:.1},{:.1}", robot_cx_cm, robot_cy_cm);

            let cx_px = (robot_cx_cm / px_to_cm_x) as i32;
            let cy_px = (robot_cy_cm / px_to_cm_y) as i32;
            let radius_px = (DETECTION_RADIUS_CM / px_to_cm_x) as i32; 
            imgproc::circle(&mut frame, Point::new(cx_px, cy_px), radius_px, Scalar::new(0.0, 255.0, 0.0, 0.0), 1, imgproc::LINE_8, 0)?;
        }

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

        let obstacles_packet = if obstacles_vector.is_empty() { "OB:none".to_string() } else { format!("OB:{}", obstacles_vector.join("|")) };
        let final_packet = format!("{};{}\n", robot_packet, obstacles_packet);
        let _ = socket.send_to(final_packet.as_bytes(), robot_ip);

        imgproc::put_text(&mut frame, &final_packet.trim(), Point::new(10, 30), imgproc::FONT_HERSHEY_SIMPLEX, 0.5, Scalar::new(0.0, 255.0, 255.0, 0.0), 1, imgproc::LINE_8, false)?;
        
        let safe_zone = Rect::new(45, 45, frame_width as i32 - 90, frame_height as i32 - 90);
        imgproc::rectangle(&mut frame, safe_zone, Scalar::new(0.0, 100.0, 255.0, 0.0), 1, imgproc::LINE_8, 0)?;

        highgui::imshow("Brain Tracker", &frame)?;
        highgui::imshow("Debug: Combined Mask", &combined_mask)?; // Смотрим на итоговую склеенную маску!

        if highgui::wait_key(1)? == 113 { break; }
    }
    Ok(())
}