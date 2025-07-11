import os
import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import warnings
from collections import defaultdict
from scipy.ndimage import gaussian_filter

# Отключаем все предупреждения
warnings.filterwarnings('ignore')

def create_paper_texture(size, noise_intensity=0.1):
    """Создает текстуру бумаги с различными оттенками белого"""
    # Создаем базовый светлый фон (слегка кремовый оттенок)
    base_color = np.random.uniform(0.92, 0.98)  # Базовый оттенок белого
    texture = np.ones((*size, 3), dtype=np.float32) * base_color
    
    # Добавляем легкий оттенок желтого/кремового
    yellow_tint = np.random.uniform(0, 0.02)  # Очень легкий желтый оттенок
    texture[:, :, 0] *= (1.0 - yellow_tint)  # Немного уменьшаем синий канал
    texture[:, :, 1] *= (1.0 - yellow_tint * 0.5)  # Чуть меньше уменьшаем зеленый
    
    # Добавляем текстуру бумаги (мелкие неровности)
    fine_noise = np.random.randn(*size) * noise_intensity * 0.5
    fine_noise = gaussian_filter(fine_noise, sigma=0.5)
    fine_noise = fine_noise[:, :, np.newaxis]
    
    # Добавляем более крупные неровности
    coarse_noise = np.random.randn(*size) * noise_intensity
    coarse_noise = gaussian_filter(coarse_noise, sigma=2.0)
    coarse_noise = coarse_noise[:, :, np.newaxis]
    
    # Комбинируем шумы
    combined_noise = (fine_noise + coarse_noise) * 0.5
    texture += np.repeat(combined_noise, 3, axis=2)
    
    # Добавляем легкие вертикальные или горизонтальные полосы (имитация структуры бумаги)
    if random.random() < 0.5:  # 50% шанс на вертикальные полосы
        stripe_noise = np.random.randn(1, size[1]) * noise_intensity * 0.3
        stripe_noise = np.repeat(stripe_noise, size[0], axis=0)
    else:  # горизонтальные полосы
        stripe_noise = np.random.randn(size[0], 1) * noise_intensity * 0.3
        stripe_noise = np.repeat(stripe_noise, size[1], axis=1)
    stripe_noise = gaussian_filter(stripe_noise, sigma=3.0)
    texture += np.dstack([stripe_noise, stripe_noise, stripe_noise])
    
    # Нормализуем значения и конвертируем в uint8
    texture = np.clip(texture * 255, 192, 255).astype(np.uint8)
    
    return texture

def generate_random_color():
    """Генерирует случайный цвет для ручки в формате BGR"""
    # Создаем разные оттенки путем варьирования всех компонентов
    variations = [
        lambda: np.array([180, 20, 5]),     # Синий
        lambda: np.array([107, 46, 9]),     # Темно-синий #1
        lambda: np.array([82, 33, 4]),      # Темно-синий #2
        lambda: np.array([33, 13, 0]),      # Черный
        # lambda: np.array([255, 30, 10]),    # Ярко-синий
        # lambda: np.array([255, 140, 50]),   # Голубой
        lambda: np.array([200, 30, 50]),     # Синий с фиолетовым
        # lambda: np.array([220, 120, 30]),    # Морской синий
        lambda: np.array([230, 50, 20]),     # Кобальтовый синий
        lambda: np.array([190, 40, 60]),     # Индиго
        # lambda: np.array([210, 90, 40]),     # Сапфировый
        lambda: np.array([250, 10, 5]),      # Ультрамарин
        lambda: np.array([147, 62, 66]),     # Синяя пыль
        lambda: np.array([177, 102, 115]),   # Крайола
        lambda: np.array([148, 64, 67]),     # Глубокий фиолетовый
    ]
    
    # Выбираем базовый цвет
    base_color = variations[np.random.randint(0, len(variations))]()
    
    # Добавляем небольшую случайность для вариативности
    noise = np.random.uniform(-20, 20, 3)
    color = np.clip(base_color + noise, 0, 255)
    
    return color / 255.0

def apply_pen_effect_classic_blue(digit_image, color=None):
    """Классический синий цвет ручки - тонкие четкие линии"""
    if color is None:
        color = np.array([200, 20, 0]) / 255.0  # Дефолтный цвет
    else:
        color = color / 255.0 if color.max() > 1 else color
        
    # Уменьшаем толщину линий с помощью эрозии
    kernel = np.ones((2, 2), np.uint8)
    digit_eroded = cv2.erode(digit_image, kernel, iterations=1)
    # digit_eroded = digit_image
    
    digit_mask = (digit_eroded == 0).astype(np.float32)
    
    # Делаем линии тоньше
    digit_mask = gaussian_filter(digit_mask, sigma=0.2)
    digit_mask = np.power(digit_mask, 2.0)
    
    # Минимальная текстура для классической ручки
    stroke_texture = gaussian_filter(np.random.randn(*digit_mask.shape) * 0.01, sigma=0.2)
    digit_mask += (digit_mask > 0.1) * stroke_texture
    
    # Создаем градиентную маску краев с несколькими уровнями размытия
    edge_mask1 = gaussian_filter(digit_mask, sigma=0.5)
    edge_mask2 = gaussian_filter(digit_mask, sigma=1.0)
    edge_mask3 = gaussian_filter(digit_mask, sigma=2.0)
    
    # Комбинируем маски для создания градиентного перехода
    edge_gradient = (edge_mask1 - edge_mask2) * 0.6 + (edge_mask2 - edge_mask3) * 0.4
    edge_gradient = np.clip(edge_gradient * 2.0, 0, 1)  # Усиливаем эффект
    
    # Применяем градиентную прозрачность
    alpha = digit_mask * (1.0 - edge_gradient)
    alpha = np.clip(alpha, 0, 1)
    
    # Дополнительно усиливаем контраст в центре линий
    alpha = np.where(alpha > 0.5, alpha * 1.2, alpha * 0.8)
    alpha = np.clip(alpha, 0, 1)
    
    result = np.ones((*digit_mask.shape, 3), dtype=np.float32)
    for i in range(3):
        result[:, :, i] = 1.0 - alpha * (1.0 - color[i])
    
    return (result * 255).astype(np.uint8)

def apply_pen_effect_gel(digit_image, color=None):
    """Эффект гелевой ручки - гладкие линии с сильным блеском"""
    if color is None:
        color = np.array([255, 20, 5]) / 255.0  # Дефолтный цвет
    else:
        color = color / 255.0 if color.max() > 1 else color
        
    digit_mask = (digit_image == 0).astype(np.float32)
    
    # Делаем линии тоньше, но с плавными краями
    digit_mask = gaussian_filter(digit_mask, sigma=0.3)
    digit_mask = np.power(digit_mask, 1.8)
    
    # Создаем градиентную маску краев с несколькими уровнями размытия
    edge_mask1 = gaussian_filter(digit_mask, sigma=0.3)
    edge_mask2 = gaussian_filter(digit_mask, sigma=0.8)
    edge_mask3 = gaussian_filter(digit_mask, sigma=1.5)
    
    # Комбинируем маски для создания градиентного перехода
    edge_gradient = (edge_mask1 - edge_mask2) * 0.7 + (edge_mask2 - edge_mask3) * 0.3
    edge_gradient = np.clip(edge_gradient * 2.5, 0, 1)  # Усиливаем эффект
    
    # Применяем градиентную прозрачность с сохранением яркости в центре
    alpha = digit_mask * (1.0 - edge_gradient * 0.8)
    alpha = np.clip(alpha, 0, 1)
    
    # Усиливаем контраст в центре для более яркого эффекта
    alpha = np.where(alpha > 0.6, alpha * 1.3, alpha * 0.7)
    alpha = np.clip(alpha, 0, 1)
    
    result = np.ones((*digit_mask.shape, 3), dtype=np.float32)
    for i in range(3):
        result[:, :, i] = 1.0 - alpha * (1.0 - color[i])
    
    # Добавляем блики с учетом новой маски прозрачности
    gloss_mask = (alpha > 0.7) * (np.random.rand(*digit_mask.shape) < 0.03)
    gloss = gaussian_filter(gloss_mask.astype(float), sigma=0.2) * 0.4
    result += np.dstack([gloss] * 3)
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)

def apply_pen_effect_fountain(digit_image, color=None):
    """Эффект перьевой ручки - характерные утолщения и растекания"""
    if color is None:
        color = np.array([180, 30, 40]) / 255.0  # Дефолтный цвет
    else:
        color = color / 255.0 if color.max() > 1 else color
        
    digit_mask = (digit_image == 0).astype(np.float32)
    
    # Базовая линия с небольшим размытием
    digit_mask = gaussian_filter(digit_mask, sigma=0.3)
    
    # Создаем эффект неравномерного нажима
    pressure = np.random.randn(*digit_mask.shape) * 0.15 + 1.0
    pressure = gaussian_filter(pressure, sigma=1.5)
    digit_mask *= pressure
    
    # Добавляем характерные утолщения в местах изменения направления
    gradient_x = gaussian_filter(digit_mask, sigma=1.0, order=[1,0])
    gradient_y = gaussian_filter(digit_mask, sigma=1.0, order=[0,1])
    gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Усиливаем эффект в местах с большим градиентом
    thickening = gradient_mag * 0.5
    thickening = gaussian_filter(thickening, sigma=0.5)
    digit_mask = np.maximum(digit_mask, thickening)
    
    # Эффект растекания чернил
    ink_spread = gaussian_filter(digit_mask, sigma=0.7)
    digit_mask = np.maximum(digit_mask, ink_spread * 0.3)
    
    # Создаем градиентную маску краев с учетом растекания
    edge_mask1 = gaussian_filter(digit_mask, sigma=0.5)
    edge_mask2 = gaussian_filter(digit_mask, sigma=1.2)
    edge_mask3 = gaussian_filter(digit_mask, sigma=2.5)
    
    # Комбинируем маски для создания градиентного перехода
    edge_gradient = (edge_mask1 - edge_mask2) * 0.5 + (edge_mask2 - edge_mask3) * 0.5
    edge_gradient = np.clip(edge_gradient * 3.0, 0, 1)  # Усиливаем эффект
    
    # Добавляем случайные вариации в прозрачность для имитации неравномерного растекания
    variation = gaussian_filter(np.random.randn(*digit_mask.shape) * 0.1, sigma=1.0)
    edge_gradient += variation * (edge_gradient > 0.2)
    edge_gradient = np.clip(edge_gradient, 0, 1)
    
    # Применяем градиентную прозрачность с учетом растекания
    alpha = digit_mask * (1.0 - edge_gradient)
    alpha = np.clip(alpha, 0, 1)
    
    # Усиливаем контраст в центре для более насыщенного эффекта
    alpha = np.where(alpha > 0.5, alpha * 1.4, alpha * 0.6)
    alpha = np.clip(alpha, 0, 1)
    
    # Финальное сглаживание
    alpha = gaussian_filter(alpha, sigma=0.2)
    
    result = np.ones((*digit_mask.shape, 3), dtype=np.float32)
    for i in range(3):
        result[:, :, i] = 1.0 - alpha * (1.0 - color[i])
    
    return (result * 255).astype(np.uint8)

def apply_pen_effect_ballpoint(digit_image, color=None):
    """Эффект шариковой ручки - прерывистые линии с характерными пробелами"""
    if color is None:
        color = np.array([210, 35, 15]) / 255.0  # Дефолтный цвет
    else:
        color = color / 255.0 if color.max() > 1 else color
        
    digit_mask = (digit_image == 0).astype(np.float32)
    
    # Базовая тонкая линия
    digit_mask = gaussian_filter(digit_mask, sigma=0.2)
    digit_mask = np.power(digit_mask, 1.8)
    
    # Создаем эффект неравномерного нажима
    pressure = np.random.randn(*digit_mask.shape) * 0.1 + 0.9
    pressure = gaussian_filter(pressure, sigma=1.0)
    digit_mask *= pressure
    
    # Добавляем микротекстуру вместо полос
    texture = np.random.randn(*digit_mask.shape) * 0.15
    texture = gaussian_filter(texture, sigma=0.3)
    digit_mask *= (1.0 + texture * (digit_mask > 0.1))
    
    # Небольшие случайные пропуски
    gaps = np.random.rand(*digit_mask.shape) > 0.98
    gaps = gaussian_filter(gaps.astype(float), sigma=0.2) * 0.3
    digit_mask *= (1.0 - gaps)
    
    # Создаем градиентную маску краев с резкими переходами
    edge_mask1 = gaussian_filter(digit_mask, sigma=0.3)
    edge_mask2 = gaussian_filter(digit_mask, sigma=0.8)
    edge_mask3 = gaussian_filter(digit_mask, sigma=1.5)
    
    # Комбинируем маски для создания градиентного перехода с акцентом на резкость
    edge_gradient = (edge_mask1 - edge_mask2) * 0.8 + (edge_mask2 - edge_mask3) * 0.2
    edge_gradient = np.clip(edge_gradient * 2.0, 0, 1)  # Сначала клипим значения
    edge_gradient = np.power(np.maximum(edge_gradient, 0), 1.5)  # Затем применяем степень к неотрицательным значениям
    
    # Добавляем дополнительную текстуру к градиенту
    edge_texture = gaussian_filter(np.random.randn(*digit_mask.shape) * 0.1, sigma=0.2)
    edge_gradient += edge_texture * (edge_gradient > 0.2)
    edge_gradient = np.clip(edge_gradient, 0, 1)
    
    # Применяем градиентную прозрачность
    alpha = digit_mask * (1.0 - edge_gradient)
    
    # Усиливаем контраст для более четких линий
    alpha = np.where(alpha > 0.4, alpha * 1.5, alpha * 0.5)
    alpha = np.clip(alpha, 0, 1)
    
    result = np.ones((*digit_mask.shape, 3), dtype=np.float32)
    for i in range(3):
        result[:, :, i] = 1.0 - alpha * (1.0 - color[i])
    
    # Убеждаемся, что все значения в допустимом диапазоне перед конвертацией
    result = np.clip(result * 255, 0, 255)
    return result.astype(np.uint8)

def apply_pen_effect(digit_image, pressure_variation=True):
    """Применяет случайный эффект ручки к изображению"""
    # Список доступных эффектов
    effects = [
        apply_pen_effect_classic_blue,
        apply_pen_effect_gel,
        apply_pen_effect_fountain,
        apply_pen_effect_ballpoint
    ]
    
    # Выбираем случайный эффект
    effect_func = random.choice(effects)
    
    # Генерируем случайный цвет
    color = generate_random_color()
    
    # Применяем эффект
    return effect_func(digit_image, color)

class TouchingDigitsLoader:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.numbers = []
        self.labels = []
        self.usage_count = defaultdict(int)
        
        # Загружаем все числа из датасета
        print("Загрузка чисел из SyntheticDigitStrings...")
        number_dirs = list(self.dataset_path.iterdir())
        for number_dir in tqdm(number_dirs, desc="Загрузка директорий"):
            if not number_dir.is_dir():
                continue
                
            number = number_dir.name
            txt_files = list(number_dir.glob('*.txt'))
            if not txt_files:
                continue
                
            # Берем только один файл из каждой директории для ускорения
            txt_file = random.choice(txt_files)
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                if len(lines) < 2:  # Пропускаем файлы без разметки
                    continue
                
                # Получаем изображение
                img_path = txt_file.with_suffix('.png')
                if not img_path.exists():
                    continue
                    
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                
                # Преобразуем изображение в оттенки серого
                if len(img.shape) == 3:
                    if img.shape[2] == 4:  # RGBA
                        # Используем RGB каналы для получения изображения в оттенках серого
                        rgb = img[:, :, :3]
                        digit = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                    else:  # RGB
                        digit = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:  # Уже в оттенках серого
                    digit = img
                
                # Нормализуем значения и убеждаемся, что цифры черные (0) на белом фоне (255)
                digit = cv2.normalize(digit, None, 0, 255, cv2.NORM_MINMAX)
                
                # Если цифры светлые на темном фоне, инвертируем
                if np.mean(digit[digit > 127]) < np.mean(digit[digit <= 127]):
                    digit = cv2.bitwise_not(digit)
                
                # Улучшаем контраст
                digit = cv2.threshold(digit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                # Парсим разметку
                boxes = []
                for line in lines[2:]:  # Пропускаем первые две строки
                    parts = line.strip().split('|')
                    if len(parts) != 4:
                        continue
                    class_id = int(parts[0])
                    coords = parts[1].split(',') + parts[2].split(',')
                    boxes.append([int(x) for x in coords] + [class_id])
                
                self.numbers.append((digit, boxes, number))
                self.labels.append(number)
        
        if not self.numbers:
            raise ValueError(f"No numbers found in {dataset_path}")
        
        print(f"Загружено {len(self.numbers)} чисел из SyntheticDigitStrings")
    
    def get_number(self):
        """Получить случайное число из датасета"""
        if not self.numbers:
            return None
            
        # Выбираем число, которое использовалось меньше всего
        idx = min(range(len(self.numbers)), key=lambda x: self.usage_count[x])
        self.usage_count[idx] += 1
        
        # Сбрасываем счетчики, если все числа использовались слишком много раз
        if min(self.usage_count.values()) > 20:
            self.usage_count.clear()
            
        return self.numbers[idx]

class LetterNoiseLoader:
    def __init__(self, letters_path):
        """Загружает буквы для использования в качестве шума"""
        self.letters_path = Path(letters_path)
        self.letters = []
        self.letter_classes = []
        
        # Создаем список допустимых папок (буквы a-z и A-Z)
        valid_folders = []
        for folder in self.letters_path.iterdir():
            if folder.is_dir() and len(folder.name) == 1:
                if folder.name.isalpha():  # Только буквы
                    valid_folders.append(folder.name)
        
        print(f"Найдено {len(valid_folders)} папок с буквами: {sorted(valid_folders)}")
        
        # Загружаем изображения букв
        for folder_name in valid_folders:
            folder_path = self.letters_path / folder_name
            for img_file in folder_path.glob("*.png"):
                try:
                    # Загружаем изображение с альфа-каналом
                    img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        # Если изображение имеет альфа-канал (4 канала)
                        if img.shape[-1] == 4:
                            # Берем только альфа-канал
                            alpha = img[:, :, 3]
                            # Создаем белое изображение
                            letter = np.ones_like(alpha) * 255
                            # Где альфа > 0, там ставим 0 (черный)
                            letter[alpha > 0] = 0
                        else:
                            # Если нет альфа-канала, используем как есть
                            letter = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        self.letters.append(letter)
                        self.letter_classes.append(folder_name)
                except Exception as e:
                    print(f"Ошибка при загрузке {img_file}: {e}")
        
        print(f"Загружено {len(self.letters)} букв для шума")
    
    def get_random_letter(self):
        """Возвращает случайную букву"""
        if not self.letters:
            return None, None
        idx = random.randint(0, len(self.letters) - 1)
        return self.letters[idx], self.letter_classes[idx]

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union

def is_valid_overlap(box1, box2, max_iou=0.15):
    """Check if the overlap between two boxes is valid"""
    iou = calculate_iou(box1, box2)
    return iou < max_iou

def check_alignment(boxes, tolerance=5):
    """Проверяет, что все цифры находятся на одной линии"""
    if not boxes:
        return True
    # Берем среднюю линию каждого bounding box
    mid_lines = [(box[1] + box[3]) / 2 for box in boxes]
    # Проверяем, что все средние линии находятся в пределах tolerance от первой
    reference = mid_lines[0]
    return all(abs(mid - reference) <= tolerance for mid in mid_lines)

def draw_pen_line(image, start_point, end_point, color, thickness=2):
    """Рисует линию в стиле ручки с неравномерностями и эффектами"""
    # Создаем пустое изображение для линии
    line_image = np.ones_like(image) * 255
    
    # Рисуем базовую линию
    cv2.line(line_image, start_point, end_point, (0, 0, 0), thickness)
    
    # Конвертируем в оттенки серого
    if len(line_image.shape) == 3:
        line_gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    else:
        line_gray = line_image
    
    # Применяем эффект ручки
    return apply_pen_effect_fountain(line_gray, color)

def add_noise_and_lines(image, boxes, probability=1.0):
    """Add random noise and lines that intersect with digits from below"""
    if random.random() > probability:
        return image
    
    noise_layer = np.zeros_like(image)
    
    if boxes:
        min_x = min(box[0] for box in boxes)
        max_x = max(box[2] for box in boxes)
        min_y = min(box[1] for box in boxes)
        max_y = max(box[3] for box in boxes)
        
        # Add horizontal lines under digits
        if random.random() < 0.5:
            num_h_lines = random.randint(2, 4)
            for _ in range(num_h_lines):
                extension = random.randint(20, 50)
                line_start_x = max(0, min_x - extension)
                line_end_x = min(image.shape[1], max_x + extension)
                y_pos = int(max_y + random.randint(5, 20))
                
                num_points = random.randint(4, 7)
                points = []
                for i in range(num_points):
                    x = line_start_x + (line_end_x - line_start_x) * i / (num_points - 1)
                    y = y_pos + random.randint(-2, 2)
                    points.append((int(x), int(y)))
                
                # Генерируем случайный цвет для линии
                color = generate_random_color() * 255
                
                # Рисуем сегменты линии
                for i in range(len(points) - 1):
                    # Получаем сегмент линии в стиле ручки
                    line_segment = draw_pen_line(
                        noise_layer, 
                        points[i], 
                        points[i + 1],
                        color,
                        thickness=random.randint(1, 2)
                    )
                    
                    # Определяем область для вставки сегмента
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    min_x = max(0, min(x1, x2) - 5)
                    max_x = min(image.shape[1], max(x1, x2) + 5)
                    min_y = max(0, min(y1, y2) - 5)
                    max_y = min(image.shape[0], max(y1, y2) + 5)
                    
                    # Создаем маску для текущего сегмента
                    segment_mask = np.mean(line_segment[min_y:max_y, min_x:max_x], axis=2) < 250
                    segment_mask = segment_mask.astype(np.float32)
                    
                    # Накладываем сегмент на noise_layer
                    noise_layer[min_y:max_y, min_x:max_x] = (
                        noise_layer[min_y:max_y, min_x:max_x] * (1 - segment_mask[:, :, np.newaxis]) +
                        line_segment[min_y:max_y, min_x:max_x] * segment_mask[:, :, np.newaxis]
                    )
    
    # Добавляем линии на изображение
    mask = np.any(noise_layer > 0, axis=2).astype(np.float32)
    mask = mask[:, :, np.newaxis]
    mask = np.repeat(mask, 3, axis=2)
    # Немного уменьшаем интенсивность линий
    noise_layer = noise_layer * 0.9
    image = image * (1 - mask) + noise_layer * mask
    
    return image

def get_bounding_box(image):
    """Get bounding box coordinates for non-zero pixels with padding 2-4 pixels"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    coords = np.argwhere(gray == 0)
    if len(coords) == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    padding = random.randint(2, 4)
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image.shape[1] - 1, x_max + padding)
    y_max = min(image.shape[0] - 1, y_max + padding)
    return [x_min, y_min, x_max, y_max]

def convert_to_yolo_format(box, image_width, image_height):
    """Convert bounding box to YOLO format [x_center, y_center, width, height]"""
    x_min, y_min, x_max, y_max = box
    
    # Calculate center coordinates and dimensions
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    
    return [x_center, y_center, width, height]

class DigitSelector:
    def __init__(self, dataset_path):
        self.digits = []
        self.labels = []
        self.usage_count = defaultdict(int)
        self.class_indices = defaultdict(list)
        
        # Load digits from dataset
        for digit_class in range(10):
            digit_dir = os.path.join(dataset_path, str(digit_class), str(digit_class))
            if not os.path.exists(digit_dir):
                print(f"Warning: Directory {digit_dir} does not exist")
                continue
                
            for img_name in os.listdir(digit_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(digit_dir, img_name)
                    # Загружаем изображение с альфа-каналом
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        # Если изображение имеет альфа-канал (4 канала)
                        if len(img.shape) == 3 and img.shape[2] == 4:
                            # Создаем маску из альфа-канала
                            alpha = img[:, :, 3]
                            # Нормализуем альфа-канал
                            alpha = cv2.normalize(alpha, None, 0, 255, cv2.NORM_MINMAX)
                            # Инвертируем маску (черные цифры на белом фоне)
                            digit = cv2.bitwise_not(alpha)
                        else:
                            # Если нет альфа-канала, преобразуем в оттенки серого
                            digit = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            # Инвертируем если нужно (предполагаем, что цифры светлые на темном фоне)
                            if np.mean(digit) > 127:
                                digit = cv2.bitwise_not(digit)
                        
                        # Убеждаемся, что цифры черные на белом фоне
                        if np.mean(digit[digit > 0]) < 127:
                            digit = cv2.bitwise_not(digit)
                        
                        # Нормализуем значения
                        digit = cv2.normalize(digit, None, 0, 255, cv2.NORM_MINMAX)
                        
                        self.digits.append(digit)
                        self.labels.append(digit_class)
                        self.class_indices[digit_class].append(len(self.digits) - 1)
        
        if not self.digits:
            raise ValueError(f"No images found in {dataset_path}. Please check the dataset path and image files.")
        
        print(f"Loaded {len(self.digits)} digits in total")
        for i in range(10):
            print(f"Class {i}: {len(self.class_indices[i])} digits")
    
    def get_digit(self, class_id=None):
        if class_id is not None:
            available_indices = [idx for idx in self.class_indices[class_id] 
                               if self.usage_count[idx] < 20]
            if not available_indices:
                for idx in self.class_indices[class_id]:
                    self.usage_count[idx] = 0
                available_indices = self.class_indices[class_id]
        else:
            # Выбираем случайный класс цифры
            available_classes = list(range(10))
            random.shuffle(available_classes)
            for class_id in available_classes:
                available_indices = [idx for idx in self.class_indices[class_id] 
                                   if self.usage_count[idx] < 20]
                if available_indices:
                    break
            if not available_indices:
                self.usage_count.clear()
                available_indices = self.class_indices[class_id]
        
        if not available_indices:
            raise ValueError("No available digits found. This should not happen if the dataset was loaded correctly.")
        
        idx = min(available_indices, key=lambda x: self.usage_count[x])
        self.usage_count[idx] += 1
        return self.digits[idx], self.labels[idx]

def add_letter_noise(image, letter_loader, boxes, num_letters=10, probability=0.9):
    """Добавляет буквенные шумы на изображение того же размера, что и цифры"""
    if random.random() > probability or not letter_loader:
        return image
    
    # Определяем количество букв для добавления
    num_to_add = random.randint(num_letters, num_letters + 10)  # 10-20 букв
    
    # Получаем размеры изображения
    h, w = image.shape[:2]
    
    # Определяем размер букв на основе размера цифр
    if boxes:
        # Вычисляем средний размер цифр
        digit_heights = []
        for box in boxes:
            digit_height = box[3] - box[1]  # y2 - y1
            digit_heights.append(digit_height)
        
        if digit_heights:
            avg_digit_height = sum(digit_heights) / len(digit_heights)
            letter_height = int(avg_digit_height * 1.2)  # Буквы на 20% больше цифр
        else:
            letter_height = int(h * 0.12)  # Fallback размер
    else:
        letter_height = int(h * 0.12)  # Fallback размер
    
    # Определяем несколько возможных горизонталей для букв
    possible_lines = []
    if boxes:
        min_y = min(box[1] for box in boxes)
        max_y = max(box[3] for box in boxes)
        h_step = int(letter_height * 1.1)
        # Строка над цифрами
        if min_y - h_step >= 0:
            possible_lines.append(max(0, min_y - h_step))
        # Строка с цифрами (основная)
        possible_lines.append(min_y)
        # Строка под цифрами
        if max_y + h_step < h:
            possible_lines.append(min(h - letter_height, max_y + h_step))
        # Если места много, добавим ещё одну строку выше и ниже
        if min_y - 2 * h_step >= 0:
            possible_lines.append(max(0, min_y - 2 * h_step))
        if max_y + 2 * h_step < h:
            possible_lines.append(min(h - letter_height, max_y + 2 * h_step))
    else:
        # Если нет цифр, размещаем в центре
        possible_lines = [int((h - letter_height) // 2)]
    
    # Создаем список уже занятых областей для избежания перекрытий
    occupied_areas = []
    added_count = 0
    
    for _ in range(num_to_add):
        # Получаем случайную букву
        letter_img, letter_class = letter_loader.get_random_letter()
        if letter_img is None:
            continue

        # Вычисляем ширину буквы с сохранением пропорций
        aspect_ratio = letter_img.shape[1] / letter_img.shape[0]
        width = int(letter_height * aspect_ratio)

        letter_img = cv2.resize(letter_img, (width, letter_height))
        
        # Применяем эффект ручки к букве
        colored_letter = apply_pen_effect(letter_img)
        
        # Создаем маску из цветного изображения
        letter_mask = np.mean(colored_letter, axis=2) < 250
        letter_mask = letter_mask.astype(np.float32)
        
        # Изменяем размер буквы и маски
        # resized_letter = cv2.resize(colored_letter, (width, letter_height))
        # resized_mask = cv2.resize(letter_mask, (width, letter_height))
        resized_letter = colored_letter
        resized_mask = letter_mask
        
        # Размещаем букву в выбранной области
        max_attempts = 50
        for attempt in range(max_attempts):
            # Случайная позиция в выбранном диапазоне
            x = random.randint(0, w - width)
            # Выбираем случайную строку для размещения буквы
            letter_y = random.choice(possible_lines)
            
            # Проверяем границы
            if x < 0 or x + width > w or letter_y < 0 or letter_y + letter_height > h:
                continue
            
            # Проверяем перекрытие с цифрами
            letter_area = [x, letter_y, x + width, letter_y + letter_height]
            overlap_with_digits = False
            
            for box in boxes:
                if not (letter_area[2] < box[0] or letter_area[0] > box[2] or 
                       letter_area[3] < box[1] or letter_area[1] > box[3]):
                    overlap_with_digits = True
                    break
            
            if overlap_with_digits:
                continue
            
            # Проверяем перекрытие с уже размещенными буквами
            overlap_with_letters = False
            for area in occupied_areas:
                if not (letter_area[2] < area[0] or letter_area[0] > area[2] or 
                       letter_area[3] < area[1] or letter_area[1] > area[3]):
                    overlap_with_letters = True
                    break
            
            if not overlap_with_letters:
                # Размещаем букву
                roi = image[letter_y:letter_y + letter_height, x:x + width]
                # Расширяем маску до 3 каналов
                mask = np.dstack([resized_mask] * 3)
                # Смешиваем с фоном
                roi[:] = roi * (1 - mask) + resized_letter * mask
                
                # Добавляем область в список занятых
                occupied_areas.append(letter_area)
                added_count += 1
                break
    
    return image

def apply_high_quality_resize(digit_image, target_height, max_rotation=5):
    """
    Применяет высококачественное масштабирование к изображению цифры с случайными искажениями
    """
    # Определяем размеры с сохранением пропорций
    aspect_ratio = digit_image.shape[1] / digit_image.shape[0]
    target_width = int(target_height * aspect_ratio)
    
    # Добавляем отступы для поворота
    padding = int(target_height * 0.2)
    padded_height = target_height + 2 * padding
    padded_width = target_width + 2 * padding
    
    # Выбираем метод интерполяции в зависимости от типа масштабирования
    scale_factor = target_height / digit_image.shape[0]
    if scale_factor > 1:
        interpolation = cv2.INTER_CUBIC  # Лучше для увеличения
    else:
        interpolation = cv2.INTER_AREA   # Лучше для уменьшения
    
    # Масштабируем изображение
    resized = cv2.resize(digit_image, (target_width, target_height), interpolation=interpolation)
    
    # Создаем изображение с отступами
    padded = np.ones((padded_height, padded_width), dtype=np.uint8) * 255
    padded[padding:padding+target_height, padding:padding+target_width] = resized
    
    # С вероятностью 60% применяем наклон вправо на 2-7 градусов
    if random.random() < 0.8:
        angle = random.uniform(8, 28)  # Положительный угол для наклона вправо
        rotation_matrix = cv2.getRotationMatrix2D(
            (padded_width/2, padded_height/2), -angle, 1.0)  # Отрицательный угол для наклона вправо
        rotated = cv2.warpAffine(padded, rotation_matrix, (padded_width, padded_height),
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=[255])
    else:
        rotated = padded
    
    # Добавляем легкое перспективное искажение
    if random.random() < 0.3:  # 30% шанс
        height, width = rotated.shape
        # Создаем точки в правильном формате для OpenCV
        src_points = np.array([
            [0, 0],
            [width-1, 0],
            [0, height-1],
            [width-1, height-1]
        ], dtype=np.float32)
        
        # Небольшие случайные смещения углов
        max_shift = width * 0.05
        dst_points = src_points + np.random.uniform(-max_shift, max_shift, src_points.shape).astype(np.float32)
        
        # Применяем перспективное преобразование
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(rotated, transform_matrix, (width, height),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=[255])
    else:
        warped = rotated
    
    # Обрезаем отступы, находя границы цифры
    coords = cv2.findNonZero(255 - warped)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = warped[y:y+h, x:x+w]
        
        # Масштабируем до целевой высоты
        final_width = int(target_height * (cropped.shape[1] / cropped.shape[0]))
        final = cv2.resize(cropped, (final_width, target_height), interpolation=cv2.INTER_CUBIC)
    else:
        final = cv2.resize(warped, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    # Применяем легкое размытие для устранения артефактов
    final = cv2.GaussianBlur(final, (3, 3), 0.5)
    
    return final

def apply_color_to_digit(digit_image, color):
    """
    Применяет случайный эффект ручки к изображению
    digit_image: изображение в оттенках серого, где цифры черные (0) на белом фоне (255)
    color: BGR цвет в формате [B, G, R], каждый канал от 0 до 255
    """
    # Выбираем случайный эффект
    effects = [
        apply_pen_effect_classic_blue,
        apply_pen_effect_gel,
        apply_pen_effect_fountain,
        apply_pen_effect_ballpoint
    ]
    effect_func = random.choice(effects)

    _, gray_image = cv2.threshold(digit_image, 128, 255, cv2.THRESH_BINARY)
    
    # Применяем эффект
    return effect_func(gray_image, color)

def generate_multi_digit_image(digit_selector, touching_loader, target_size=640, min_digits=5, max_digits=10, max_attempts=50):
    """Генерирует изображение с цифрами и их масками"""
    for attempt in range(max_attempts):
        # Создаем фон как текстуру бумаги
        image = create_paper_texture((target_size, target_size))
        
        boxes = []  # [x1, y1, x2, y2, class_id]
        masks = []  # [(mask, class_id), ...]
        base_height = int(target_size * 0.12)
        
        # Генерируем только из single digits
        if digit_selector is not None:
            num_digits = random.randint(min_digits, max_digits)
            
            # Собираем информацию о всех цифрах
            digit_info = []
            total_width = 0
            for _ in range(num_digits):
                digit, label = digit_selector.get_digit()
                aspect_ratio = digit.shape[1] / digit.shape[0]
                width = int(base_height * aspect_ratio)
                resized_digit = cv2.resize(digit, (width, base_height),
                                          interpolation=cv2.INTER_LANCZOS4)
                
                # Применяем цвет к цифрам
                color = generate_random_color() * 255
                colored_digit = apply_color_to_digit(resized_digit, color)
                
                # Создаем маску из цветного изображения
                digit_mask = np.mean(colored_digit, axis=2) < 250
                digit_mask = digit_mask.astype(np.float32)
                
                digit_info.append((width, base_height, colored_digit, digit_mask, label))
                total_width += width
            
            # Базовый отступ между цифрами
            spacing = int(base_height * 0.1)
            total_width += spacing * (num_digits - 1)
            
            if total_width > target_size * 0.9:
                continue
            
            # Центрируем по горизонтали и вертикали
            start_x = (target_size - total_width) // 2
            start_y = (target_size - base_height) // 2
            
            # Размещаем цифры
            current_x = start_x
            for width, height, colored_digit, digit_mask, label in digit_info:
                # Создаем полную маску для текущей цифры
                full_mask = np.zeros((target_size, target_size), dtype=np.float32)
                full_mask[start_y:start_y + height, current_x:current_x + width] = digit_mask
                
                # Размещаем цифру на изображении
                roi = image[start_y:start_y + height, current_x:current_x + width]
                roi[:] = roi * (1 - digit_mask[:, :, np.newaxis]) + colored_digit * digit_mask[:, :, np.newaxis]
                
                boxes.append([current_x, start_y, current_x + width, start_y + height, label])
                masks.append((full_mask, label))
                current_x += width + spacing
        
        if len(boxes) >= min_digits:
            return image, boxes, masks
    
    print("Не удалось разместить все цифры корректно за несколько попыток")
    return image, boxes, []

def convert_mask_to_yolo_segments(mask, image_width, image_height):
    """Конвертирует маску в формат сегментации YOLO (список нормализованных координат контура)"""
    # Находим контуры маски
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Берем самый большой контур
    contour = max(contours, key=cv2.contourArea)
    
    # Упрощаем контур для уменьшения количества точек
    epsilon = 0.005 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    
    # Преобразуем контур в нормализованные координаты
    contour = contour.reshape(-1, 2)
    normalized_contour = contour.astype(np.float32)
    normalized_contour[:, 0] = normalized_contour[:, 0] / image_width
    normalized_contour[:, 1] = normalized_contour[:, 1] / image_height
    
    return normalized_contour.flatten().tolist()

def main():
    # ============================================================================
    # КОНФИГУРАЦИЯ ГЕНЕРАЦИИ ДАТАСЕТА
    # ============================================================================
    
    # а) Использовать ли датасет Touching Digits
    USE_TOUCHING_DIGITS = False  # Отключаем использование touching digits
    TOUCHING_DIGITS_PATH = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/Datasets/Touching-Digits/SyntheticDigitStrings"
    
    # б) Использовать ли цифры для этого датасета (Single Digits)
    USE_SINGLE_DIGITS = True
    SINGLE_DIGITS_PATH = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/Datasets/Single-Digits/dataset"
    
    # в) Добавлять ли разметку для датасета с буквами (системно без разметки)
    ADD_LETTER_ANNOTATIONS = False  # Буквы добавляются как шум без разметки
    
    # г) Параметры эффекта ручки
    PEN_PRESSURE_VARIATION = True  # Включить вариацию давления ручки
    PAPER_TEXTURE_INTENSITY = 0.08  # Интенсивность текстуры бумаги (уменьшили)
    
    # ============================================================================
    # ПАРАМЕТРЫ ГЕНЕРАЦИИ
    # ============================================================================
    num_images = 10000  # Количество генерируемых изображений
    target_size = 640  # Размер изображения
    min_digits = 5    # Минимальное количество цифр
    max_digits = 10   # Максимальное количество цифр
    
    # ============================================================================
    
    # Create output directories
    output_dir = Path("/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/yolo-dataset-v11-seg")
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize digit selector
    digit_selector = None
    if USE_SINGLE_DIGITS:
        try:
            digit_selector = DigitSelector(SINGLE_DIGITS_PATH)
            print(f"Загрузчик одиночных цифр инициализирован: {SINGLE_DIGITS_PATH}")
        except Exception as e:
            print(f"Ошибка при инициализации загрузчика одиночных цифр: {e}")
            USE_SINGLE_DIGITS = False
    
    # Initialize touching digits loader
    touching_loader = None
    if USE_TOUCHING_DIGITS:
        try:
            touching_loader = TouchingDigitsLoader(TOUCHING_DIGITS_PATH)
            print(f"Загрузчик touching digits инициализирован: {TOUCHING_DIGITS_PATH}")
        except Exception as e:
            print(f"Ошибка при инициализации загрузчика touching digits: {e}")
            USE_TOUCHING_DIGITS = False
    
    # Initialize letter noise loader
    letters_path = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/Datasets/Hnd-Letters/Img"
    try:
        letter_loader = LetterNoiseLoader(letters_path)
        print("Загрузчик буквенных шумов инициализирован успешно")
    except Exception as e:
        print(f"Ошибка при инициализации загрузчика букв: {e}")
        letter_loader = None
    
    # Проверяем, что хотя бы один загрузчик цифр работает
    if not USE_SINGLE_DIGITS and not USE_TOUCHING_DIGITS:
        raise ValueError("Необходимо включить хотя бы один источник цифр (USE_SINGLE_DIGITS или USE_TOUCHING_DIGITS)")
    
    # Generate images
    for i in tqdm(range(num_images)):
        image, boxes, masks = generate_multi_digit_image(digit_selector, touching_loader, target_size, min_digits, max_digits)
        
        # Добавляем шум и линии к изображению
        image = add_noise_and_lines(image, boxes)
        
        # Добавляем буквенные шумы
        image = add_letter_noise(image, letter_loader, boxes)
        
        # Save image
        image_path = images_dir / f"image_{i:04d}.jpg"
        cv2.imwrite(str(image_path), image)
        
        # Save labels in YOLO format with segmentation
        label_path = labels_dir / f"image_{i:04d}.txt"
        with open(label_path, 'w') as f:
            for mask, class_id in masks:
                # Конвертируем маску в формат сегментации YOLO
                segments = convert_mask_to_yolo_segments(mask, image.shape[1], image.shape[0])
                if segments:
                    # Записываем в формате: class_id x1 y1 x2 y2 ... xn yn
                    f.write(f"{class_id} {' '.join(map(str, segments))}\n")

if __name__ == "__main__":
    main()  