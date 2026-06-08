use opencv::{
    core::{self, Point, Scalar, Vector},
    highgui, imgproc, prelude::*, videoio, Result as CvResult,
};
use std::net::UdpSocket;
use std::time::Duration;

// Функция для поиска цветной метки
fn get_marker_data(
    frame: &core::Mat,
    lower_bound: core::Scalar,
    upper_bound: core::Scalar,
) -> CvResult<Option<(Point, f64)>> {
    let mut hsv = core::Mat::default();
    imgproc::cvt_color_def(frame, &mut hsv, imgproc::COLOR_BGR2HSV)?;

    let mut mask = core::Mat::default();
    core::in_range(&hsv, &lower_bound, &upper_bound, &mut mask)?;

    let kernel = core::Mat::default();
    let mut eroded = core::Mat::default();
    imgproc::erode(&mask, &mut eroded, &kernel, Point::new(-1, -1), 2, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;
    let mut dilated = core::Mat::default();
    imgproc::dilate(&eroded, &mut dilated, &kernel, Point::new(-1, -1), 2, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;

    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours(&dilated, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

    let mut max_area = 0.0;
    let mut best_idx = -1;

    for i in 0..contours.len() {
        let area = imgproc::contour_area(&contours.get(i)?, false)?;
        if area > max_area && area > 100.0 {
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
    // --- НАСТРОЙКА СЕТИ ---
    // Создаем UDP сокет на порту 8888 нашего Мака
    let socket = UdpSocket::bind("0.0.0.0:8888").expect("Не удалось привязать сокет");
    // ЗДЕСЬ НУЖНО БУДЕТ УКАЗАТЬ IP-АДРЕС РОБОТА В WI-FI СЕТИ
    let robot_ip = "192.168.1.100:9999"; 
    println!("UDP Передатчик запущен. Отправка команд на {}", robot_ip);

    // --- НАСТРОЙКА КАМЕРЫ ---
    // --- НАСТРОЙКА КАМЕРЫ ---
    let mut cap = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    
    // Проверяем, отдала ли система камеру
    if !cap.is_opened()? {
        println!("❌ ОШИБКА: система не дала доступ к камере! Проверь настройки 'Конфиденциальность и безопасность'.");
        return Ok(());
    }
    println!("✅ Камера успешно подключена!");

    let mut frame = core::Mat::default();

    // Цвета меток (откалибруй под свое освещение)
    let lower_green = Scalar::new(40.0, 100.0, 100.0, 0.0);
    let upper_green = Scalar::new(80.0, 255.0, 255.0, 0.0);
    let lower_pink = Scalar::new(160.0, 100.0, 100.0, 0.0);
    let upper_pink = Scalar::new(180.0, 255.0, 255.0, 0.0);

    let mut base_distance: Option<f64> = None;
    let base_speed = 50;
    let slope_threshold = 15.0;

    loop {
        cap.read(&mut frame)?;
        if frame.empty() { 
            println!("⚠️ Камера отдала пустой кадр, ждем...");
            continue; // Не выходим, а просто пробуем прочитать следующий кадр
        }

        let front_data = get_marker_data(&frame, lower_green, upper_green)?;
        let rear_data = get_marker_data(&frame, lower_pink, upper_pink)?;

        let mut current_speed = base_speed;

        if let (Some((front_pos, front_area)), Some((rear_pos, rear_area))) = (front_data, rear_data) {
            imgproc::circle(&mut frame, front_pos, 10, Scalar::new(0.0, 255.0, 0.0, 0.0), -1, imgproc::LINE_8, 0)?;
            imgproc::circle(&mut frame, rear_pos, 10, Scalar::new(255.0, 192.0, 203.0, 0.0), -1, imgproc::LINE_8, 0)?;
            imgproc::line(&mut frame, front_pos, rear_pos, Scalar::new(255.0, 255.0, 255.0, 0.0), 2, imgproc::LINE_8, 0)?;

            let dx = (rear_pos.x - front_pos.x) as f64;
            let dy = (rear_pos.y - front_pos.y) as f64;
            let current_distance = (dx * dx + dy * dy).sqrt();

            if base_distance.is_none() {
                base_distance = Some(current_distance);
                println!("Калибровка завершена: {:.1} px", current_distance);
            }

            if let Some(bd) = base_distance {
                if current_distance < (bd - slope_threshold) {
                    if front_area > rear_area * 1.1 {
                        current_speed = base_speed + 30; // Подъем
                    } else if rear_area > front_area * 1.1 {
                        current_speed = base_speed - 20; // Спуск
                    }
                }
            }

            // --- ОТПРАВКА КОМАНДЫ ПО WI-FI ---
            let command = format!("CMD:{}\n", current_speed);
            // Пытаемся отправить, игнорируем ошибку, если робот еще не включен
            let _ = socket.send_to(command.as_bytes(), robot_ip);

            // Вывод на экран
            let text = format!("SPEED: {}", current_speed);
            imgproc::put_text(&mut frame, &text, Point::new(10, 30), imgproc::FONT_HERSHEY_SIMPLEX, 0.9, Scalar::new(0.0, 255.0, 255.0, 0.0), 2, imgproc::LINE_8, false)?;
        }

        highgui::imshow("Mac M5 Brain Tracker", &frame)?;

        let key = highgui::wait_key(1)?;
        if key == 113 { // 'q' для выхода
            break;
        } else if key == 99 { // 'c' для сброса калибровки
            base_distance = None;
        }
    }

    Ok(())
}