use opencv::{
    core::{self, Point, Rect, Scalar, Vector},
    highgui, imgproc, prelude::*, videoio, Result as CvResult,
};
use std::net::UdpSocket;

const ARENA_LENGTH_CM: f64 = 142.0; 
const ARENA_WIDTH_CM: f64 = 77.0;   

const MIN_OBSTACLE_AREA: f64 = 15.0; 
const MAX_OBSTACLE_AREA: f64 = 140.0;
// Снизил порог площади робота на случай, если камера висит выше обычного
const MIN_ROBOT_AREA: f64 = 60.0; 
const MAX_ROBOT_AREA: f64 = 700.0;
const DETECTION_RADIUS_CM: f64 = 20.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let socket = UdpSocket::bind("0.0.0.0:8888").expect("Не удалось привязать сокет");
    let robot_ip = "192.168.1.107:8888"; 
    
    println!("========================================");
    println!("🛠️ РЕЖИМ ЖИВОЙ КАЛИБРОВКИ ЗАПУЩЕН!");
    println!("Кликни по окну с видео и используй кнопки:");
    println!(" [W] - Увеличить лимит Яркости (V)");
    println!(" [S] - Уменьшить лимит Яркости (V)");
    println!(" [E] - Увеличить лимит Цвета (S)");
    println!(" [D] - Уменьшить лимит Цвета (S)");
    println!(" [Q] - Выход");
    println!("========================================");

    let mut cap_opt = None;
    for index in 0..6 {
        if let Ok(try_cap) = videoio::VideoCapture::new(index, videoio::CAP_ANY) {
            if let Ok(opened) = try_cap.is_opened() {
                if opened { cap_opt = Some(try_cap); break; }
            }
        }
    }

    let mut cap = match cap_opt {
        Some(c) => c,
        None => { println!("❌ ОШИБКА: Камера не найдена."); return Ok(()); }
    };

    let frame_width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as f64;
    let frame_height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as f64;
    let px_to_cm_x = ARENA_LENGTH_CM / frame_width;
    let px_to_cm_y = ARENA_WIDTH_CM / frame_height;
    let px2_to_cm2 = px_to_cm_x * px_to_cm_y; 

    highgui::named_window("Brain Tracker", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("Debug: Mask", highgui::WINDOW_AUTOSIZE)?;
    
    let mut frame = core::Mat::default();

    // Стартовые значения
    let mut v_max = 160.0; 
    let mut s_max = 85.0;  

    loop {
        cap.read(&mut frame)?;
        if frame.empty() { continue; }

        let mut blurred = core::Mat::default();
        imgproc::gaussian_blur_def(&frame, &mut blurred, core::Size::new(7, 7), 0.0)?;

        let mut hsv = core::Mat::default();
        imgproc::cvt_color_def(&blurred, &mut hsv, imgproc::COLOR_BGR2HSV)?;

        // Фильтр обновляется в реальном времени!
        let lower_black = Scalar::new(0.0, 0.0, 0.0, 0.0);
        let upper_black = Scalar::new(180.0, s_max, v_max, 0.0); 

        let mut black_mask = core::Mat::default();
        core::in_range(&hsv, &lower_black, &upper_black, &mut black_mask)?;

        let kernel = core::Mat::default();
        let mut temp_mask = core::Mat::default();
        imgproc::dilate(&black_mask, &mut temp_mask, &kernel, Point::new(-1, -1), 6, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;
        imgproc::erode(&temp_mask, &mut black_mask, &kernel, Point::new(-1, -1), 2, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;

        let mut contours = Vector::<Vector<Point>>::new();
        imgproc::find_contours(&mut black_mask, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

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

            // РИСУЕМ ЖЕЛТЫЕ РАМКИ ВОКРУГ ВСЕГО НАЙДЕННОГО, ЧТОБЫ ТЫ ВИДЕЛ ПЛОЩАДЬ
            if area_cm2 > 10.0 {
                imgproc::rectangle(&mut frame, rect, Scalar::new(0.0, 255.0, 255.0, 0.0), 1, imgproc::LINE_8, 0)?;
                imgproc::put_text(&mut frame, &format!("{:.0}cm2", area_cm2), Point::new(rect.x, rect.y - 15), imgproc::FONT_HERSHEY_SIMPLEX, 0.4, Scalar::new(0.0, 255.0, 255.0, 0.0), 1, imgproc::LINE_8, false)?;
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

            imgproc::rectangle(&mut frame, r_rect, Scalar::new(0.0, 255.0, 0.0, 0.0), 3, imgproc::LINE_8, 0)?;
            imgproc::put_text(&mut frame, "ROBOT", Point::new(r_rect.x, r_rect.y - 5), imgproc::FONT_HERSHEY_SIMPLEX, 0.6, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, false)?;

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
                } 
            } 
        }

        let obstacles_packet = if obstacles_vector.is_empty() { "OB:none".to_string() } else { format!("OB:{}", obstacles_vector.join("|")) };
        let final_packet = format!("{};{}\n", robot_packet, obstacles_packet);
        let _ = socket.send_to(final_packet.as_bytes(), robot_ip);

        // ВЫВОД ЗНАЧЕНИЙ НА ЭКРАН
        imgproc::put_text(&mut frame, &format!("V Max (Brightness): {}", v_max), Point::new(10, 30), imgproc::FONT_HERSHEY_SIMPLEX, 0.7, Scalar::new(0.0, 0.0, 255.0, 0.0), 2, imgproc::LINE_8, false)?;
        imgproc::put_text(&mut frame, &format!("S Max (Color): {}", s_max), Point::new(10, 60), imgproc::FONT_HERSHEY_SIMPLEX, 0.7, Scalar::new(0.0, 0.0, 255.0, 0.0), 2, imgproc::LINE_8, false)?;
        
        let safe_zone = Rect::new(45, 45, frame_width as i32 - 90, frame_height as i32 - 90);
        imgproc::rectangle(&mut frame, safe_zone, Scalar::new(0.0, 100.0, 255.0, 0.0), 1, imgproc::LINE_8, 0)?;

        highgui::imshow("Brain Tracker", &frame)?;
        highgui::imshow("Debug: Mask", &black_mask)?;

        // ОБРАБОТКА НАЖАТИЙ КЛАВИШ
        let key = highgui::wait_key(1)?;
        if key == 113 { break; } // q
        if key == 119 { v_max = (v_max + 10.0).min(255.0); } // w (Увеличить V)
        if key == 115 { v_max = (v_max - 10.0).max(0.0);   } // s (Уменьшить V)
        if key == 101 { s_max = (s_max + 5.0).min(255.0);  } // e (Увеличить S)
        if key == 100 { s_max = (s_max - 5.0).max(0.0);    } // d (Уменьшить S)
    }
    Ok(())
}