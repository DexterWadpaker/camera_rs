use opencv::{
    core::{self, Point, Rect, Scalar, Vector},
    highgui, imgproc, prelude::*, videoio, Result as CvResult,
};
use std::net::UdpSocket;

// Размеры стенда в сантиметрах
const ARENA_LENGTH_CM: f64 = 142.0; 
const ARENA_WIDTH_CM: f64 = 77.0;   

// --- НАСТРОЙКА ПЛОЩАДЕЙ (Калибруй эти числа по экрану!) ---
// Для кубика (базовый 72.2 см^2): пусть ловит в диапазоне от 30 до 140
const MIN_OBSTACLE_AREA: f64 = 30.0;
const MAX_OBSTACLE_AREA: f64 = 140.0;

// Для робота (базовый 340 см^2): расширяем нижний порог, так как на полу он кажется меньше
const MIN_ROBOT_AREA: f64 = 150.0; // Если на полу всё равно не видит, опусти до 100
const MAX_ROBOT_AREA: f64 = 500.0;

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
        if area > max_area && area > 15.0 {
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
    let robot_ip = "192.168.1.107:8888"; 
    println!("📡 Умный трекер запущен. Анализ геометрии стенда...");

    let mut cap = videoio::VideoCapture::new(0, videoio::CAP_V4L2)?;
    if !cap.is_opened()? {
        println!("❌ ОШИБКА: Камера недоступна.");
        return Ok(());
    }

    let frame_width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as f64;
    let frame_height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as f64;
    
    let px_to_cm_x = ARENA_LENGTH_CM / frame_width;
    let px_to_cm_y = ARENA_WIDTH_CM / frame_height;
    let px2_to_cm2 = px_to_cm_x * px_to_cm_y; 

    highgui::named_window("Brain Tracker", highgui::WINDOW_AUTOSIZE)?;
    
    let mut frame = core::Mat::default();
    let mut black_mask = core::Mat::default();

    // --- АНТИ-ТЕНЬ НАСТРОЙКИ (HSV) ---
    // Чтобы отсечь тени, мы уменьшаем верхний порог Насыщенности (S) с 255 до 80.
    // Тень на зеленом поле будет иметь высокую насыщенность зеленого, а робот — низкую.
    let lower_black = Scalar::new(0.0, 0.0, 0.0, 0.0);
    let upper_black = Scalar::new(180.0, 80.0, 70.0, 0.0); // S < 80 (нет цвета), V < 70 (темно)

    // Цвета маркеров направления на крыше робота
    let lower_blue = Scalar::new(100.0, 100.0, 50.0, 0.0);
    let upper_blue = Scalar::new(140.0, 255.0, 255.0, 0.0);
    let lower_pink = Scalar::new(155.0, 50.0, 50.0, 0.0);
    let upper_pink = Scalar::new(185.0, 255.0, 255.0, 0.0);

    loop {
        cap.read(&mut frame)?;
        if frame.empty() { continue; }

        let mut hsv = core::Mat::default();
        imgproc::cvt_color_def(&frame, &mut hsv, imgproc::COLOR_BGR2HSV)?;
        core::in_range(&hsv, &lower_black, &upper_black, &mut black_mask)?;

        // Фильтры очистки маски
        let kernel = core::Mat::default();
        let mut temp_mask = core::Mat::default();
        imgproc::erode(&black_mask, &mut temp_mask, &kernel, Point::new(-1, -1), 1, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;
        imgproc::dilate(&temp_mask, &mut black_mask, &kernel, Point::new(-1, -1), 2, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;

        let mut contours = Vector::<Vector<Point>>::new();
        imgproc::find_contours(&mut black_mask, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

        let mut obstacles_vector = Vec::new();
        let mut robot_packet = "RB:none".to_string();
        let mut robot_rect_px: Option<Rect> = None;

        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area_px = imgproc::contour_area(&contour, false)?;
            let area_cm2 = area_px * px2_to_cm2;

            let rect = imgproc::bounding_rect(&contour)?;

            // ИГНОРИРУЕМ СОВСЕМ МЕЛКИЙ МУСОР (меньше 15 см^2)
            if area_cm2 < 15.0 { continue; }

            // Классификация 1: Препятствие
            if area_cm2 >= MIN_OBSTACLE_AREA && area_cm2 <= MAX_OBSTACLE_AREA {
                let cx = (rect.x + rect.width / 2) as f64 * px_to_cm_x;
                let cy = (rect.y + rect.height / 2) as f64 * px_to_cm_y;
                obstacles_vector.push(format!("{:.1},{:.1}", cx, cy));
                imgproc::rectangle(&mut frame, rect, Scalar::new(0.0, 0.0, 255.0, 0.0), 2, imgproc::LINE_8, 0)?;
                imgproc::put_text(&mut frame, &format!("{:.0} cm2", area_cm2), Point::new(rect.x, rect.y - 5), imgproc::FONT_HERSHEY_SIMPLEX, 0.4, Scalar::new(0.0, 0.0, 255.0, 0.0), 1, imgproc::LINE_8, false)?;
            
            // Классификация 2: Робот
            } else if area_cm2 >= MIN_ROBOT_AREA && area_cm2 <= MAX_ROBOT_AREA {
                robot_rect_px = Some(rect);
                let cx = (rect.x + rect.width / 2) as f64 * px_to_cm_x;
                let cy = (rect.y + rect.height / 2) as f64 * px_to_cm_y;
                robot_packet = format!("RB:{:.1},{:.1}", cx, cy);
                imgproc::put_text(&mut frame, &format!("ROBOT: {:.0} cm2", area_cm2), Point::new(rect.x, rect.y - 5), imgproc::FONT_HERSHEY_SIMPLEX, 0.5, Scalar::new(0.0, 255.0, 0.0, 0.0), 1, imgproc::LINE_8, false)?;
            
            // Отладка: Если объект не подошел ни под что, рисуем его серым и пишем его площадь!
            } else {
                imgproc::rectangle(&mut frame, rect, Scalar::new(128.0, 128.0, 128.0, 0.0), 1, imgproc::LINE_8, 0)?;
                imgproc::put_text(&mut frame, &format!("? {:.0} cm2", area_cm2), Point::new(rect.x, rect.y + 15), imgproc::FONT_HERSHEY_SIMPLEX, 0.4, Scalar::new(128.0, 128.0, 128.0, 0.0), 1, imgproc::LINE_8, false)?;
            }
        }

        // Поиск цветных меток направления
        let front_marker = get_orientation_marker(&frame, lower_blue, upper_blue)?;
        let rear_marker = get_orientation_marker(&frame, lower_pink, upper_pink)?;

        if let (Some(r_rect), Some(front), Some(rear)) = (robot_rect_px, front_marker, rear_marker) {
            imgproc::rectangle(&mut frame, r_rect, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0)?;
            imgproc::line(&mut frame, rear, front, Scalar::new(255.0, 255.0, 255.0, 0.0), 2, imgproc::LINE_8, 0)?;
        } else if let Some(r_rect) = robot_rect_px {
            imgproc::rectangle(&mut frame, r_rect, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0)?;
        }

        let obstacles_packet = if obstacles_vector.is_empty() { "OB:none".to_string() } else { format!("OB:{}", obstacles_vector.join("|")) };
        let final_packet = format!("{};{}\n", robot_packet, obstacles_packet);
        let _ = socket.send_to(final_packet.as_bytes(), robot_ip);

        imgproc::put_text(&mut frame, &final_packet.trim(), Point::new(10, 30), imgproc::FONT_HERSHEY_SIMPLEX, 0.5, Scalar::new(0.0, 255.0, 255.0, 0.0), 1, imgproc::LINE_8, false)?;
        highgui::imshow("Brain Tracker", &frame)?;

        if highgui::wait_key(1)? == 113 { break; }
    }
    Ok(())
}