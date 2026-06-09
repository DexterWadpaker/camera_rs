use opencv::{
    core::{self, Point, Scalar, Vector},
    highgui, imgproc, prelude::*, videoio, Result as CvResult,
};
use std::net::UdpSocket;

// Функция поиска метки теперь также возвращает свою черно-белую маску для отладки
fn get_marker_data(
    frame: &core::Mat,
    lower_bound: core::Scalar,
    upper_bound: core::Scalar,
    out_mask: &mut core::Mat, // Сюда будем записывать маску
) -> CvResult<Option<(Point, f64)>> {
    let mut hsv = core::Mat::default();
    imgproc::cvt_color_def(frame, &mut hsv, imgproc::COLOR_BGR2HSV)?;

    let mut raw_mask = core::Mat::default();
    core::in_range(&hsv, &lower_bound, &upper_bound, &mut raw_mask)?;

    let kernel = core::Mat::default();
    let mut eroded = core::Mat::default();
    imgproc::erode(&raw_mask, &mut eroded, &kernel, Point::new(-1, -1), 2, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;
    
    // Результат дилатации сразу пишем в out_mask, чтобы показать его на экране
    imgproc::dilate(&eroded, out_mask, &kernel, Point::new(-1, -1), 2, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;

    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours(out_mask, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

    let mut max_area = 0.0;
    let mut best_idx = -1;

    for i in 0..contours.len() {
        let area = imgproc::contour_area(&contours.get(i)?, false)?;
        // Снизили порог со 100 до 30 пикселей, чтобы камера видела даже мелкие метки
        if area > max_area && area > 30.0 {
            max_area = area;
            best_idx = i as i32;
        }
    }

    if best_idx >= 0 {
        let moments = imgproc::moments(&contours.get(best_idx as usize)?, false)?;
        if moments.m00 != 0.0 {
            let cx = (moments.m10 / moments.m00) as i32;
            let cy = (moments.m01 / moments.m00) as i32;
            return Ok(Some((Point::new(cx, cy), max_area)));
        }
    }
    Ok(None)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let socket = UdpSocket::bind("0.0.0.0:8888").expect("Не удалось привязать сокет");
    let robot_ip = "192.168.1.107:8888"; 
    println!("UDP Передатчик запущен. Отправка команд на {}", robot_ip);

    let mut cap = videoio::VideoCapture::new(0, videoio::CAP_AVFOUNDATION)?;
    if !cap.is_opened()? {
        println!("❌ ОШИБКА: Камера недоступна.");
        return Ok(());
    }

    // Создаем три окна (Главное + Две маски)
    highgui::named_window("Brain Tracker", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("Debug: Blue Mask", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("Debug: Pink Mask", highgui::WINDOW_AUTOSIZE)?;

    let mut frame = core::Mat::default();

    // --- РАСШИРЕННЫЕ НАСТРОЙКИ ЦВЕТОВ ---
    // Мы опустили порог насыщенности (S) и яркости (V) со 100 до 50, чтобы ловить цвета в тени
    let lower_blue = Scalar::new(100.0, 100.0, 50.0, 0.0);
    let upper_blue = Scalar::new(140.0, 255.0, 255.0, 0.0);
    
    let lower_pink = Scalar::new(155.0, 50.0, 50.0, 0.0);
    let upper_pink = Scalar::new(185.0, 255.0, 255.0, 0.0);

    let mut base_distance: Option<f64> = None;
    let base_speed = 50;
    let slope_threshold = 15.0;

    // Матрицы для хранения черно-белых картинок
    let mut blue_mask = core::Mat::default();
    let mut pink_mask = core::Mat::default();

    loop {
        cap.read(&mut frame)?;
        if frame.empty() { continue; }

        let front_data = get_marker_data(&frame, lower_blue, upper_blue, &mut blue_mask)?;
        let rear_data = get_marker_data(&frame, lower_pink, upper_pink, &mut pink_mask)?;

        let mut current_speed = base_speed;

        if let (Some((front_pos, front_area)), Some((rear_pos, rear_area))) = (front_data, rear_data) {
            // Синий круг на носу: B=255, G=0, R=0
            imgproc::circle(&mut frame, front_pos, 10, Scalar::new(255.0, 0.0, 0.0, 0.0), -1, imgproc::LINE_8, 0)?;
            // Розовый круг на корме
            imgproc::circle(&mut frame, rear_pos, 10, Scalar::new(255.0, 192.0, 203.0, 0.0), -1, imgproc::LINE_8, 0)?;
            // Линия между ними
            imgproc::line(&mut frame, front_pos, rear_pos, Scalar::new(255.0, 255.0, 255.0, 0.0), 2, imgproc::LINE_8, 0)?;

            let dx = (rear_pos.x - front_pos.x) as f64;
            let dy = (rear_pos.y - front_pos.y) as f64;
            let current_distance = (dx * dx + dy * dy).sqrt();

            if base_distance.is_none() {
                base_distance = Some(current_distance);
                println!("Калибровка: {:.1} px", current_distance);
            }

            if let Some(bd) = base_distance {
                if current_distance < (bd - slope_threshold) {
                    if front_area > rear_area * 1.1 {
                        current_speed = base_speed + 30; 
                    } else if rear_area > front_area * 1.1 {
                        current_speed = base_speed - 20; 
                    }
                }
            }

            let command = format!("CMD:{}\n", current_speed);
            let _ = socket.send_to(command.as_bytes(), robot_ip);

            let text = format!("SPEED: {}", current_speed);
            imgproc::put_text(&mut frame, &text, Point::new(10, 30), imgproc::FONT_HERSHEY_SIMPLEX, 0.9, Scalar::new(0.0, 255.0, 255.0, 0.0), 2, imgproc::LINE_8, false)?;
        }

        // Показываем все три окна
        highgui::imshow("Brain Tracker", &frame)?;
        highgui::imshow("Debug: Blue Mask", &blue_mask)?;
        highgui::imshow("Debug: Pink Mask", &pink_mask)?;

        let key = highgui::wait_key(1)?;
        if key == 113 { // 'q'
            break;
        } else if key == 99 { // 'c'
            base_distance = None;
        }
    }

    Ok(())
}