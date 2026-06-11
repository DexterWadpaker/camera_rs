use opencv::{
    core::{self, Point, Rect, Scalar, Vector},
    highgui, imgproc, prelude::*, videoio, Result as CvResult,
};
use std::net::UdpSocket;

// Размеры стенда (сантиметры)
const ARENA_LENGTH_CM: f64 = 142.0; // Ось X
const ARENA_WIDTH_CM: f64 = 77.0;   // Ось Y

// Ожидаемые площади объектов (см^2)
const OBSTACLE_AREA_CM2: f64 = 8.5 * 8.5; // ~72.2
const ROBOT_AREA_CM2: f64 = 20.0 * 17.0;  // ~340.0
// Допустимая погрешность площади (в процентах, т.к. ракурс может искажать)
const AREA_TOLERANCE_PCT: f64 = 0.40; // +/- 40%

// Функция поиска центра цветного маркера (для ориентации робота)
fn get_marker_point(
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
        if area > max_area && area > 20.0 { // Метки мелкие, порог ниже
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
    // Настройка сети
    let socket = UdpSocket::bind("0.0.0.0:8888").expect("Не удалось привязать сокет");
    let robot_ip = "192.168.1.100:9999"; // IP Raspberry Pi на роботе
    println!("📡 Геометрический трекер запущен. Данные на {}", robot_ip);

    // Подключение камеры (Ubuntu, V4L2, пробуем индекс 0 или 1)
    let mut cap = videoio::VideoCapture::new(0, videoio::CAP_V4L2)?; 
    if !cap.is_opened()? {
        println!("❌ ОШИБКА: Камера на индексе 0 недоступна.");
        return Ok(());
    }

    // Коэффициенты масштаба (см/пиксель)
    let frame_width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as f64;
    let frame_height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as f64;
    let px_to_cm_x = ARENA_LENGTH_CM / frame_width;
    let px_to_cm_y = ARENA_WIDTH_CM / frame_height;
    // Средний коэффициент для расчета площади
    let px2_to_cm2 = px_to_cm_x * px_to_cm_y; 

    println!("📐 Камера: {:.0}x{:.0} px. Масштаб площади: 1px^2 = {:.4}см^2", 
              frame_width, frame_height, px2_to_cm2);

    highgui::named_window("Geometric Stand Tracker (Ubuntu)", highgui::WINDOW_AUTOSIZE)?;
    
    let mut frame = core::Mat::default();
    let mut black_mask = core::Mat::default();

    // --- НАСТРОЙКИ (HSV) ---
    // 1. Фильтр для ЧЁРНЫХ объектов на ЗЕЛЁНОМ фоне.
    // Мы берем весь диапазон цветов (0-180), любую насыщенность (0-255), 
    // но ограничиваем яркость (Value) низким значением, чтобы найти тёмное.
    let lower_black = Scalar::new(0.0, 0.0, 0.0, 0.0);
    let upper_black = Scalar::new(180.0, 255.0, 70.0, 0.0); // Яркость < 70 из 255

    // 2. Маркеры сверху робота (для ориентации, клеить обязательно!)
    let lower_blue = Scalar::new(100.0, 100.0, 50.0, 0.0);
    let upper_blue = Scalar::new(140.0, 255.0, 255.0, 0.0);
    let lower_pink = Scalar::new(150.0, 100.0, 50.0, 0.0);
    let upper_pink = Scalar::new(180.0, 255.0, 255.0, 0.0);

    loop {
        cap.read(&mut frame)?;
        if frame.empty() { continue; }

        // ШАГ 1: Находим все чёрные контуры
        let mut hsv = core::Mat::default();
        imgproc::cvt_color_def(&frame, &mut hsv, imgproc::COLOR_BGR2HSV)?;
        core::in_range(&hsv, &lower_black, &upper_black, &mut black_mask)?;

        // Морфология для очистки маски от шумов (теней на поле)
        let kernel = core::Mat::default();
        let mut temp_mask = core::Mat::default();
        imgproc::erode(&black_mask, &mut temp_mask, &kernel, Point::new(-1, -1), 1, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;
        imgproc::dilate(&temp_mask, &mut black_mask, &kernel, Point::new(-1, -1), 2, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;

        let mut contours = Vector::<Vector<Point>>::new();
        imgproc::find_contours(&mut black_mask, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

        let mut obstacles_vector = Vec::new();
        let mut robot_packet = "RB:none".to_string();

        // Переменные для хранения прямоугольника робота
        let mut robot_rect_px: Option<Rect> = None;

        // ШАГ 2: Классификация контуров по площади
        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area_px = imgproc::contour_area(&contour, false)?;
            let area_cm2 = area_px * px2_to_cm2; // Перевод в см^2

            // Проверка на препятствие (~72 см^2)
            if area_cm2 > OBSTACLE_AREA_CM2 * (1.0 - AREA_TOLERANCE_PCT) && 
               area_cm2 < OBSTACLE_AREA_CM2 * (1.0 + AREA_TOLERANCE_PCT) {
                
                let rect = imgproc::bounding_rect(&contour)?;
                let center_cm_x = (rect.x + rect.width / 2) as f64 * px_to_cm_x;
                let center_cm_y = (rect.y + rect.height / 2) as f64 * px_to_cm_y;
                obstacles_vector.push(format!("{:.1},{:.1}", center_cm_x, center_cm_y));

                // Отрисовка препятствия красным
                imgproc::rectangle(&mut frame, rect, Scalar::new(0.0, 0.0, 255.0, 0.0), 2, imgproc::LINE_8, 0)?;
            
            // Проверка на робота (~340 см^2)
            } else if area_cm2 > ROBOT_AREA_CM2 * (1.0 - AREA_TOLERANCE_PCT) && 
                      area_cm2 < ROBOT_AREA_CM2 * (1.0 + AREA_TOLERANCE_PCT) {
                
                let rect = imgproc::bounding_rect(&contour)?;
                robot_rect_px = Some(rect); // Сохраняем прямоугольник для отрисовки

                let center_cm_x = (rect.x + rect.width / 2) as f64 * px_to_cm_x;
                let center_cm_y = (rect.y + rect.height / 2) as f64 * px_to_cm_y;
                robot_packet = format!("RB:{:.1},{:.1}", center_cm_x, center_cm_y);
            }
        }

        // ШАГ 3: Определение ориентации робота (используем цветные метки сверху)
        let front_marker = get_marker_point(&frame, lower_blue, upper_blue)?;
        let rear_marker = get_marker_point(&frame, lower_pink, upper_pink)?;

        if let (Some(r_rect), Some(front), Some(rear)) = (robot_rect_px, front_marker, rear_marker) {
            // Отрисовка робота зелёным боксом
            imgproc::rectangle(&mut frame, r_rect, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0)?;
            // Отрисовка вектора направления
            imgproc::line(&mut frame, rear, front, Scalar::new(255.0, 255.0, 255.0, 0.0), 2, imgproc::LINE_8, 0)?;
            imgproc::circle(&mut frame, front, 5, Scalar::new(255.0, 0.0, 0.0, 0.0), -1, imgproc::LINE_8, 0)?; // Нос синий
        } else if robot_rect_px.is_some() {
             imgproc::rectangle(&mut frame, robot_rect_px.unwrap(), Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0)?;
             // Предупреждение если робот виден, а метки нет
             imgproc::put_text(&mut frame, "⚠️ NO MARKERS", Point::new(r_rect.x, r_rect.y - 5), imgproc::FONT_HERSHEY_SIMPLEX, 0.5, Scalar::new(0.0, 0.0, 255.0, 0.0), 1, imgproc::LINE_8, false)?;
        }

        // Формирование пакета
        let obstacles_packet = if obstacles_vector.is_empty() { "OB:none".to_string() } else { format!("OB:{}", obstacles_vector.join("|")) };
        let final_packet = format!("{};{}\n", robot_packet, obstacles_packet);
        let _ = socket.send_to(final_packet.as_bytes(), robot_ip);

        // Вывод на экран
        imgproc::put_text(&mut frame, &final_packet.trim(), Point::new(10, 30), imgproc::FONT_HERSHEY_SIMPLEX, 0.5, Scalar::new(0.0, 255.0, 255.0, 0.0), 1, imgproc::LINE_8, false)?;
        highgui::imshow("Geometric Stand Tracker (Ubuntu)", &frame)?;
        // Расскомментируй, чтобы видеть чёрную маску для отладки
        // highgui::imshow("Debug: Black Mask", &black_mask)?;

        if highgui::wait_key(1)? == 113 { break; } // 'q'
    }
    Ok(())
}