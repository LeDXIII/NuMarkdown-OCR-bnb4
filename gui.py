import os
import sys
import json
import traceback
import time
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QPlainTextEdit, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFileDialog, QMessageBox, QComboBox, QFrame
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, QObject, Signal, Slot, QThread

# Путь к директории с моделями
BASE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_bnb4")
# Путь к файлу с шаблонами промтов
PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "prompts.json")

class OCRWorker(QObject):
    """Рабочий поток для выполнения OCR. Обеспечивает отзывчивость интерфейса."""
    finished = Signal(str)  # Сигнал с результатом
    log = Signal(str)       # Сигнал для записи в лог

    def __init__(self, image_path, prompt_text):
        super().__init__()
        self.image_path = image_path
        self.prompt_text = prompt_text
        self.processor = None
        self.model = None

    @Slot()
    def run(self):
        """Основной метод рабочего потока. Выполняет распознавание текста на изображении."""
        try:
            if self.processor is None or self.model is None:
                self.log.emit("ОШИБКА: Модель не передана в рабочий поток!")
                self.finished.emit("Ошибка: внутренняя ошибка приложения.")
                return

            self.log.emit("Открытие изображения...")
            img = Image.open(self.image_path).convert("RGB")

            # Ресайз изображения для оптимизации скорости и потребления памяти
            max_size = 1536
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                self.log.emit(f"Изображение уменьшено до {new_size}")

            self.log.emit("Подготовка входных данных...")
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": self.prompt_text}
                    ],
                }
            ]

            # Формирование промта для модели
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Обработка изображения и подготовка тензоров
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )

            # Перемещение данных на устройство (GPU/CPU)
            inputs = inputs.to(self.model.device)
            self.log.emit(f"Input: {inputs['input_ids'].shape}, Pixels: {inputs['pixel_values'].shape}")

            self.log.emit("Начало генерации...")
            start_time = time.time()

            # Генерация текста моделью (исправлены параметры для устранения предупреждений)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    min_new_tokens=10,
                    do_sample=False,
                    repetition_penalty=1.05,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                )

            generation_time = time.time() - start_time
            self.log.emit(f"Генерация завершена за {generation_time:.1f} сек")

            # Декодирование сгенерированных токенов в читаемый текст
            generated_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]

            decoded = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            result = decoded[0].strip() if isinstance(decoded, list) and len(decoded) > 0 else ""

            # Очистка результата от обрамляющих Markdown-тегов
            if result.startswith("```markdown"):
                result = result[len("```markdown"):].lstrip()
            if result.endswith("```"):
                result = result[:-len("```")].rstrip()

            # Проверка на пустой результат
            if len(result) < 5:
                result = "Модель не смогла извлечь текст из изображения. Попробуйте другое изображение или проверьте качество."

            self.finished.emit(result)
            self.log.emit("Готово!")

        except torch.cuda.OutOfMemoryError:
            error_msg = "ОШИБКА: Недостаточно видеопамяти!"
            self.log.emit(error_msg)
            self.finished.emit(error_msg)
        except Exception as e:
            error_msg = f"Ошибка: {e}\n{traceback.format_exc()}"
            self.log.emit(error_msg)
            self.finished.emit("")

class MainWindow(QWidget):
    """Главное окно приложения. Управляет UI и логикой взаимодействия с пользователем."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NuMarkdown OCR")
        self.setGeometry(300, 200, 850, 650)

        # Инициализация переменных состояния
        self.current_model_path = None
        self.image_path = None
        self.processor = None
        self.model = None
        self.thread = None
        self.worker = None

        # Словарь шаблонов промтов будет загружен позже, после создания UI
        self.prompt_templates = {}

        # Настройка системы логирования
        self.setup_logging()

        self._build_ui()
        self.populate_model_combo()

    def load_prompts_from_file(self):
        """Загружает словарь шаблонов промтов из JSON-файла."""
        try:
            with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
                prompts = json.load(f)
            self.log(f"✅ Загружено {len(prompts)} шаблонов промтов из {PROMPTS_FILE}")
            return prompts
        except FileNotFoundError:
            error_msg = f"❌ Файл шаблонов не найден: {PROMPTS_FILE}. Создан файл по умолчанию."
            self.create_default_prompts_file()
            # Загружаем заново после создания
            with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
                prompts = json.load(f)
            self.log(error_msg)
            return prompts
        except json.JSONDecodeError as e:
            error_msg = f"❌ Ошибка чтения JSON: {e}. Создан файл по умолчанию."
            self.create_default_prompts_file()
            with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
                prompts = json.load(f)
            self.log(error_msg)
            return prompts

    def create_default_prompts_file(self):
        """Создает файл prompts.json с шаблонами по умолчанию, если он отсутствует."""
        default_prompts = {
            "Base OCR": "Please extract all text from this image and format it as clean markdown. Include all visible text, maintaining structure and formatting.",
            "Manga FULL": "Carefully and precisely extract all manga text — including character exclamations and off-panel text. Prioritize text outside page margins first. Then, extract frame text in original reading order: right-to-left, top-to-bottom. Preserve layout and sequence exactly as presented.",
            "Manga CLEAN": "Extract only the text inside manga panels — ignore all margin notes, off-panel text, and page numbers. Focus exclusively on dialogue, narration, and sound effects within frames. Present content in original reading order: right-to-left, top-to-bottom.",
            "Plain Text": "Extract text as plain, unformatted lines.",
            "Describe Layout": "Describe the content and layout of this image in detail.",
            "Body Text Only": "Extract only the main body text, ignoring headers, footers, and page numbers."
        }
        with open(PROMPTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_prompts, f, ensure_ascii=False, indent=4)

    def setup_logging(self):
        """Настраивает систему логирования: создает папку и файлы для текущего сеанса."""
        # Создаем папку logs, если ее нет
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Создаем подпапку для текущего сеанса с именем по времени запуска
        session_dir_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_dir = os.path.join(logs_dir, session_dir_name)
        os.makedirs(self.session_log_dir, exist_ok=True)

        # Пути к файлам логов
        self.info_log_path = os.path.join(self.session_log_dir, "info.log")
        self.text_log_path = os.path.join(self.session_log_dir, "text.log")

        # Инициализируем файлы (создаем пустые, если не существуют)
        open(self.info_log_path, 'a').close()
        open(self.text_log_path, 'a').close()

        # Логирование инициализации будет выполнено позже, после создания UI
        # self.log("✅ Система логирования инициализирована.")

    def _build_ui(self):
        """Создает и настраивает все элементы пользовательского интерфейса."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # Заголовок приложения
        title = QLabel("NuMarkdown OCR")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        layout.addWidget(title)

        # Верхняя панель с элементами управления
        top_panel = QGridLayout()
        top_panel.setSpacing(10)

        # Строка 1: Выбор модели и изображения
        model_label = QLabel("Модель:")
        top_panel.addWidget(model_label, 0, 0)

        self.combo_model = QComboBox()
        self.combo_model.currentIndexChanged.connect(self.on_model_changed)
        self.combo_model.setMinimumWidth(200)
        top_panel.addWidget(self.combo_model, 0, 1)

        self.btn_choose = QPushButton("📂 Выбрать изображение")
        top_panel.addWidget(self.btn_choose, 0, 2)

        self.btn_run = QPushButton("🚀 Запустить OCR")
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white;")
        top_panel.addWidget(self.btn_run, 0, 3)

        # Строка 2: Поле ввода промта и выбор шаблона
        prompt_label = QLabel("Промт:")
        top_panel.addWidget(prompt_label, 1, 0)

        self.text_prompt = QPlainTextEdit()
        self.text_prompt.setPlaceholderText("Введите промт здесь...")
        self.text_prompt.setMaximumHeight(60)
        top_panel.addWidget(self.text_prompt, 1, 1, 1, 2)

        self.combo_templates = QComboBox()
        self.combo_templates.setMaximumWidth(150)  # Ограничение ширины для компактности UI
        top_panel.addWidget(self.combo_templates, 1, 3)

        layout.addLayout(top_panel)

        # Разделительная линия
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # Панель для отображения результата
        result_panel = QVBoxLayout()
        result_panel.setSpacing(5)

        # Заголовок результата с кнопкой копирования
        result_header_layout = QHBoxLayout()
        result_label = QLabel("📝 Результат (Markdown):")
        result_header_layout.addWidget(result_label)
        result_header_layout.addStretch()

        self.btn_copy = QPushButton("📋 Копировать")
        result_header_layout.addWidget(self.btn_copy)
        result_panel.addLayout(result_header_layout)

        self.text_output = QPlainTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setStyleSheet("border: 1px solid #ccc; border-radius: 5px;")
        result_panel.addWidget(self.text_output, stretch=1)
        layout.addLayout(result_panel, stretch=3)

        # Панель лога
        log_label = QLabel("📋 Лог:")
        layout.addWidget(log_label)

        self.text_log = QPlainTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setStyleSheet("background-color: #1e1e1e; color: #dcdcdc; border: 1px solid #555; border-radius: 5px;")
        self.text_log.setMaximumHeight(120)
        layout.addWidget(self.text_log, stretch=1)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # ВАЖНО: Теперь можно безопасно загружать промты и писать в лог
        self.prompt_templates = self.load_prompts_from_file()
        # Устанавливаем промт по умолчанию (первый в словаре)
        if self.prompt_templates:
            first_key = next(iter(self.prompt_templates))
            self.text_prompt.setPlainText(self.prompt_templates[first_key])
        # Заполняем выпадающий список шаблонами
        self.combo_templates.addItems(["Выбрать шаблон..."] + list(self.prompt_templates.keys()))
        self.combo_templates.currentIndexChanged.connect(self.on_template_selected)

        # Записываем в лог, что UI готов
        self.log("✅ Система логирования инициализирована.")
        self.log("✅ Пользовательский интерфейс создан.")

        # Подключение сигналов к слотам
        self.btn_choose.clicked.connect(self.choose_image)
        self.btn_run.clicked.connect(self.start_ocr)
        self.btn_copy.clicked.connect(self.copy_result_to_clipboard)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def populate_model_combo(self):
        """Сканирует папку `model_bnb4` и заполняет выпадающий список доступными моделями."""
        self.combo_model.clear()
        try:
            if not os.path.exists(BASE_MODEL_DIR):
                self.log(f"Папка моделей не найдена: {BASE_MODEL_DIR}")
                return

            model_dirs = [d for d in os.listdir(BASE_MODEL_DIR) if os.path.isdir(os.path.join(BASE_MODEL_DIR, d))]
            if not model_dirs:
                self.log("В папке model_bnb4 не найдено моделей.")
                return

            for model_name in model_dirs:
                full_path = os.path.join(BASE_MODEL_DIR, model_name)
                self.combo_model.addItem(model_name, full_path)

            self.current_model_path = self.combo_model.itemData(0) if model_dirs else None
            self.log(f"Найдено моделей: {len(model_dirs)}")

        except Exception as e:
            self.log(f"Ошибка при сканировании моделей: {e}")

    def on_model_changed(self, index):
        """Обработчик смены выбранной модели. Обновляет путь к текущей модели."""
        model_path = self.combo_model.itemData(index)
        if model_path:
            self.current_model_path = model_path
            self.log(f"✅ Выбрана модель: {os.path.basename(model_path)}")

    def on_template_selected(self, index):
        """Обработчик выбора шаблона промта. Подставляет полный текст промта в поле ввода."""
        if index > 0:
            short_name = self.combo_templates.itemText(index)
            full_prompt = self.prompt_templates.get(short_name, "")
            self.text_prompt.setPlainText(full_prompt)

    def choose_image(self):
        """Открывает диалоговое окно для выбора изображения."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if path:
            self.image_path = path
            self.log(f"✅ Выбран файл: {os.path.basename(path)}")

    def start_ocr(self):
        """Запускает процесс OCR. Создает и запускает рабочий поток."""
        if not self.image_path:
            QMessageBox.warning(self, "Внимание", "Пожалуйста, выберите изображение.")
            return
        if not self.current_model_path:
            QMessageBox.critical(self, "Ошибка", "Модель не выбрана.")
            return

        # Загрузка модели, если она еще не загружена
        if self.processor is None or self.model is None:
            self.log("🔄 Загрузка модели...")
            self.processor, self.model = self.load_model(self.current_model_path)
            if self.processor is None or self.model is None:
                QMessageBox.critical(self, "Ошибка", "Не удалось загрузить модель.")
                return

        self.text_output.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Остановка предыдущего потока, если он еще работает
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

        prompt_text = self.text_prompt.toPlainText().strip()
        if not prompt_text:
            QMessageBox.warning(self, "Внимание", "Пожалуйста, введите промт.")
            return

        # Создание и настройка рабочего потока
        self.thread = QThread()
        self.worker = OCRWorker(self.image_path, prompt_text)
        self.worker.moveToThread(self.thread)

        # Передача модели и процессора в рабочий поток
        self.worker.processor = self.processor
        self.worker.model = self.model

        # Подключение сигналов
        self.worker.finished.connect(self.display_result)
        self.worker.log.connect(self.log)
        self.thread.started.connect(self.worker.run)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self.worker.deleteLater)

        self.thread.start()

    def copy_result_to_clipboard(self):
        """Копирует содержимое поля результата в системный буфер обмена."""
        text = self.text_output.toPlainText()
        if text.strip():
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.log("✅ Текст успешно скопирован в буфер обмена.")
        else:
            self.log("⚠️ Нет текста для копирования.")

    def display_result(self, result):
        """Отображает результат распознавания в соответствующем поле интерфейса и записывает его в text.log."""
        self.text_output.setPlainText(result)
        # Записываем результат в лог распознанного текста
        with open(self.text_log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Распознан текст:\n{result}\n{'-'*50}\n")

    def log(self, message):
        """Записывает сообщение в поле лога и в файл info.log."""
        # Вывод в UI
        timestamped_msg = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        self.text_log.appendPlainText(timestamped_msg)

        # Запись в файл info.log
        with open(self.info_log_path, 'a', encoding='utf-8') as f:
            f.write(timestamped_msg + '\n')

    def load_model(self, model_path):
        """Загружает модель и процессор с оптимизациями для 4-битного квантования."""
        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Исправлено: 'torch_dtype' -> 'dtype'
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                dtype=torch.float16,  # Исправлено: torch_dtype -> dtype
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            self.log("✅ Модель загружена успешно.")
            self.log(f"Устройство: {model.device}")
            if hasattr(model, 'quantization_method'):
                self.log(f"Метод квантования: {model.quantization_method}")

            return processor, model

        except Exception as e:
            error_msg = f"❌ Ошибка загрузки: {e}\n{traceback.format_exc()}"
            self.log(error_msg)
            return None, None

def main():
    """Точка входа в приложение. Создает и запускает главное окно."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()