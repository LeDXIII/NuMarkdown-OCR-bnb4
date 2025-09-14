# 🚀 NuMarkdown OCR

[Перейти к описанию на русском языке](#-описание-на-русском) | [Go to English description](#-english-description)

---

## 🇷🇺 Описание на русском

**NuMarkdown OCR** — это десктопное приложение для извлечения текста из изображений с помощью современных мультимодальных ИИ-моделей. Программа работает локально в формате **bnb4**, обеспечивая полную конфиденциальность ваших данных и высокую производительность даже на потребительском железе.

В отличие от классических OCR-систем, NuMarkdown OCR предлагает **гибкую настройку вывода** и **высочайшую точность распознавания**. Изначально разработанное для распознавания текста в манге и подготовки его к переводу, приложение теперь поддерживает любые VL-модели и справляется с рукописным текстом, таблицами, графиками и сложным форматированием.

---

### 🔑 Основные функции:

*   **Локальная работа**: Все вычисления происходят на вашем компьютере. Ваши изображения никогда не покидают ваше устройство.
*   **Универсальная поддержка форматов**: Извлекает текст из скриншотов, фотографий документов, страниц манги, комиксов, таблиц и графиков.
*   **Гибкие промты**: Встроенные шаблоны позволяют настроить вывод под ваши задачи — от чистого текста до сложного Markdown с сохранением структуры.
*   **Управление моделями**: Автоматически сканирует папку `model_bnb4` и позволяет выбирать модель через удобный выпадающий список.
*   **Оптимизация памяти**: Использует 4-битное квантование (bnb4) для работы с большими моделями на видеокартах с 8+ ГБ VRAM.
*   **Удобный интерфейс**: Чистый, интуитивно понятный интерфейс с логом процесса и кнопкой для копирования результата.

---

### ⚙️ Принцип работы:

1.  **Выберите изображение** через графический интерфейс.
2.  **Настройте распознавание**: Выберите модель и шаблон промта (или введите свой собственный).
3.  **Запустите OCR**: Модель проанализирует изображение и сгенерирует текст в соответствии с промтом.
4.  **Получите результат**: Текст отобразится в интерфейсе и может быть скопирован одним кликом. Все результаты автоматически сохраняются в лог-файлы.

---

### 📥 Установка и настройка:

1.  **Склонируйте репозиторий**.
2.  **Запустите `install.bat`** — скрипт автоматически создаст виртуальное окружение и установит все зависимости.  
    *Поддерживает CUDA 11.8–12.4 и Python 3.10+. Минимальная версия Python — 3.8.*
3.  **Скачайте квантованную модель** и поместите её в папку `model_bnb4`:  
    → **[LeDXIII/NuMarkdown-8B-Thinking-bnb4](https://huggingface.co/LeDXIII/NuMarkdown-8B-Thinking-bnb4)** — специально подготовленная модель для распознавания манги, рукописного текста, таблиц и графиков.
4.  **Запустите `run.bat`** — и начинайте работу!

---

## 🇬🇧 English Description

**NuMarkdown OCR** is a desktop application for extracting text from images using state-of-the-art multimodal AI models. It runs locally in **bnb4 format**, ensuring complete data privacy and high performance even on consumer-grade hardware.

Unlike traditional OCR systems, NuMarkdown OCR provides **flexible output control** and **exceptional recognition accuracy**. Originally designed for manga text extraction and translation preparation, the app now supports any VL model and handles handwritten text, tables, charts, and complex layouts with ease.

---

### 🔑 Key Features:

*   **Local Processing**: All computations happen on your machine. Your images never leave your device.
*   **Universal Format Support**: Extracts text from screenshots, document photos, manga/comic pages, tables, and charts.
*   **Flexible Prompts**: Built-in templates let you tailor output — from plain text to structured Markdown.
*   **Model Management**: Automatically scans the `model_bnb4` folder and lets you switch models via a dropdown menu.
*   **Memory Optimization**: Uses 4-bit quantization (bnb4) to run large models on GPUs with 8+ GB VRAM.
*   **User-Friendly Interface**: Clean, intuitive UI with a real-time log and one-click copy button.

---

### ⚙️ How It Works:

1.  **Select an image** via the graphical interface.
2.  **Configure recognition**: Choose a model and prompt template (or enter your own custom prompt).
3.  **Run OCR**: The model analyzes the image and generates text according to your prompt.
4.  **Get results**: Text appears in the interface and can be copied with one click. All outputs are automatically logged to files.

---

### 📥 Installation & Setup:

1.  **Clone the repository**.
2.  **Run `install.bat`** — the script will automatically set up a virtual environment and install all dependencies.  
    *Supports CUDA 11.8–12.4 and Python 3.10+. Minimum Python version: 3.8.*
3.  **Download the quantized model** and place it in the `model_bnb4` folder:  
    → **[LeDXIII/NuMarkdown-8B-Thinking-bnb4](https://huggingface.co/LeDXIII/NuMarkdown-8B-Thinking-bnb4)** — a specially prepared model for manga, handwriting, tables, and charts.
4.  **Run `run.bat`** — and start using the app!
