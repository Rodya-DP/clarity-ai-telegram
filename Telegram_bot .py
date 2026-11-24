import telebot
import requests
import time
import json
import os
import re
import threading


# --- НАСТРОЙКИ ПУТЕЙ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_FILE = os.path.join(BASE_DIR, "profiles.json")


# --- НАСТРОЙКИ TELEGRAM ---
# Можно задать BOT_TOKEN через окружение
# или заменить YOUR_TOKEN своим токеном.
BOT_TOKEN = os.environ.get("BOT_TOKEN", "YOUR_TOKEN")

LLM_URL = os.environ.get(
    "LLM_URL",
    "http://127.0.0.1:8080/v1/chat/completions"
)
MODEL_NAME = os.environ.get("MODEL_NAME", "local-model")


# --- ГЛОБАЛЬНЫЕ СОСТОЯНИЯ ---
user_profiles = {}
pending_clarifications = {}

# Защищает профили при одновременной обработке сообщений.
state_lock = threading.RLock()

# Сообщения одного чата обрабатываем по очереди.
chat_locks = {}


def load_json(filepath):
    if not os.path.exists(filepath):
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Не удалось прочитать {filepath}. "
            f"Проверь файл, чтобы не потерять память: {error}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            f"В файле {filepath} должен находиться JSON-объект."
        )

    return data


def save_json(filepath, data):
    """Сохраняет данные через временный файл."""
    temp_path = filepath + ".tmp"

    with state_lock:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        os.replace(temp_path, filepath)


def get_chat_lock(chat_id):
    with state_lock:
        if chat_id not in chat_locks:
            chat_locks[chat_id] = threading.RLock()

        return chat_locks[chat_id]


def split_text(text, limit=4000):
    parts = []

    while len(text) > limit:
        split_idx = text.rfind("\n", 0, limit)

        if split_idx <= 0:
            split_idx = text.rfind(" ", 0, limit)

        if split_idx <= 0:
            split_idx = limit

        parts.append(text[:split_idx])
        text = text[split_idx:].lstrip()

    if text:
        parts.append(text)

    return parts


def send_long_reply(message, text):
    for part in split_text(text):
        bot.reply_to(message, part)
        time.sleep(0.5)


def get_user_name(message):
    user = message.from_user

    if user is None:
        return "друг"

    if user.username:
        return f"@{user.username}"

    if user.first_name:
        return user.first_name

    return "друг"


def ensure_profile(chat_id, message=None):
    """Создаёт профиль и добавляет отсутствующие поля."""
    chat_id = str(chat_id)

    with state_lock:
        changed = False

        if chat_id not in user_profiles:
            name = get_user_name(message) if message else "Пользователь"

            user_profiles[chat_id] = {
                "identity": {
                    "name": name,
                    "gender": "",
                    "age": ""
                },
                "core": [],
                "notes": []
            }
            changed = True

        profile = user_profiles[chat_id]

        if not isinstance(profile, dict):
            raise ValueError(f"Повреждён профиль чата {chat_id}")

        defaults = {
            "identity": {
                "name": get_user_name(message) if message else "Пользователь",
                "gender": "",
                "age": ""
            },
            "core": [],
            "notes": []
        }

        for key, value in defaults.items():
            if key not in profile:
                profile[key] = value
                changed = True

        if not isinstance(profile["identity"], dict):
            raise ValueError(f"Повреждено поле identity чата {chat_id}")

        for key in ("core", "notes"):
            if not isinstance(profile[key], list):
                raise ValueError(
                    f"Поле {key} в профиле {chat_id} должно быть списком"
                )

        if changed:
            save_json(PROFILE_FILE, user_profiles)

        return profile


# --- ПОИСК ЗАМЕТОК ---
def get_relevant_notes(user_text, notes_list):
    if not notes_list:
        return "Нет заметок."

    user_words = set(
        re.findall(r"\b\w{4,}\b", user_text.lower())
    )

    relevant = []

    for note in notes_list:
        note_words = set(
            re.findall(r"\b\w{4,}\b", note.lower())
        )

        if user_words.intersection(note_words):
            relevant.append(note)

    if relevant:
        return "\n".join(relevant[-5:])

    return "Нет подходящих заметок."


# --- ПАРСИНГ ОТВЕТА МОДЕЛИ ---
# Формат: [CONFIDENCE: 85], [CORE: факт], [NOTE: заметка].
SERVICE_TAG_RE = re.compile(
    r"\[\s*(CONFIDENCE|CORE|NOTE)\s*:\s*([^\]]*)\]",
    re.IGNORECASE
)


def parse_model_reply(raw_reply):
    # Если уверенность отсутствует или повреждена,
    # не считаем ответ автоматически достоверным.
    confidence = 0
    core_memories = []
    note_memories = []

    for match in SERVICE_TAG_RE.finditer(raw_reply):
        tag = match.group(1).upper()
        value = match.group(2).strip()

        if tag == "CONFIDENCE":
            number_match = re.fullmatch(
                r"(\d{1,3})\s*%?",
                value
            )

            if number_match:
                confidence = max(
                    0,
                    min(100, int(number_match.group(1)))
                )

        elif tag == "CORE":
            memory = " ".join(value.split())

            if memory and memory not in core_memories:
                core_memories.append(memory)

        elif tag == "NOTE":
            memory = " ".join(value.split())

            if memory and memory not in note_memories:
                note_memories.append(memory)

    clean_reply = SERVICE_TAG_RE.sub("", raw_reply)
    clean_reply = re.sub(r"\n[ \t]+\n", "\n\n", clean_reply)
    clean_reply = re.sub(r"\n{3,}", "\n\n", clean_reply).strip()

    return confidence, core_memories, note_memories, clean_reply


def keep_typing(chat_id, stop_event):
    """Показывает статус «печатает», пока модель отвечает."""
    while not stop_event.is_set():
        try:
            bot.send_chat_action(chat_id, "typing")
        except Exception:
            pass

        if stop_event.wait(4):
            break


# --- ЗАГРУЗКА И ЗАПУСК КЛИЕНТА ---
user_profiles = load_json(PROFILE_FILE)

if not BOT_TOKEN or BOT_TOKEN == "YOUR_TOKEN":
    raise RuntimeError(
        "Укажи токен Telegram: замени YOUR_TOKEN в BOT_TOKEN "
        "или задай переменную окружения BOT_TOKEN."
    )

bot = telebot.TeleBot(BOT_TOKEN)


# --- КОМАНДЫ ---
@bot.message_handler(commands=["start", "reset"])
def reset_memory(message):
    chat_id = str(message.chat.id)

    with get_chat_lock(chat_id):
        pending_clarifications.pop(chat_id, None)
        ensure_profile(chat_id, message)

        user_name = get_user_name(message)

        bot.reply_to(
            message,
            f"Привет, {user_name}! Я твой ассистент. "
            "У меня нет истории сообщений, но я делаю заметки "
            "о наших разговорах и ищу по ним нужную инфу."
        )


@bot.message_handler(commands=["remember"])
def add_core_fact(message):
    chat_id = str(message.chat.id)
    command_parts = message.text.split(maxsplit=1)

    fact = (
        command_parts[1].strip()
        if len(command_parts) > 1
        else ""
    )

    if not fact:
        bot.reply_to(
            message,
            "Напиши факт после команды, например: "
            "/remember я живу в Домодедово"
        )
        return

    with get_chat_lock(chat_id):
        profile = ensure_profile(chat_id, message)

        with state_lock:
            if fact not in profile["core"]:
                profile["core"].append(fact)
                save_json(PROFILE_FILE, user_profiles)

        send_long_reply(
            message,
            f"Записал в Ядро памяти: «{fact}»"
        )


@bot.message_handler(commands=["profile"])
def show_profile(message):
    chat_id = str(message.chat.id)

    with get_chat_lock(chat_id):
        profile = ensure_profile(chat_id, message)

        with state_lock:
            core = list(profile["core"])
            notes = list(profile["notes"][-10:])

        text = "ЯДРО ПАМЯТИ:\n"
        text += "\n".join(f"• {fact}" for fact in core) if core else "Пусто"

        text += "\n\nЗАМЕТКИ (последние 10):\n"
        text += "\n".join(f"• {note}" for note in notes) if notes else "Пусто"

        send_long_reply(message, text)


# --- ОБРАБОТКА СООБЩЕНИЙ ---
@bot.message_handler(
    content_types=["text"],
    func=lambda message: True
)
def handle_message(message):
    chat_id = str(message.chat.id)

    with get_chat_lock(chat_id):
        try:
            process_message(message, chat_id)
        except Exception as error:
            print(f"[ОШИБКА] {type(error).__name__}: {error}")

            try:
                bot.reply_to(
                    message,
                    "Не удалось обработать сообщение. "
                    "Подробности ошибки записаны в консоли бота."
                )
            except Exception:
                pass


def process_message(message, chat_id):
    user_text = message.text
    profile = ensure_profile(chat_id, message)

    # 1. Уточнение предыдущего вопроса.
    clarification = pending_clarifications.get(chat_id)

    if clarification:
        original_question, bot_clarification = clarification

        final_user_prompt = (
            f"Вопрос: {original_question}\n"
            f"Мой уточняющий вопрос: {bot_clarification}\n"
            f"Ответ пользователя: {user_text}"
        )
    else:
        final_user_prompt = user_text

    # 2. Копируем память для формирования запроса.
    with state_lock:
        ident = dict(profile["identity"])
        core = list(profile["core"])
        notes = list(profile["notes"])

    ident_parts = []

    if ident.get("name"):
        ident_parts.append(f"Имя: {ident['name']}")

    if ident.get("gender"):
        ident_parts.append(f"Пол: {ident['gender']}")

    if ident.get("age"):
        ident_parts.append(f"Возраст: {ident['age']}")

    id_text = (
        "Собеседник: " + ", ".join(ident_parts) + "."
        if ident_parts
        else ""
    )

    notes_text = get_relevant_notes(final_user_prompt, notes)
    core_text = "\n".join(core) or "Нет сохранённых фактов."

    # 3. Системный промпт.
    system_prompt = (
        f"Ты — полезный ассистент. {id_text}\n"
        "Общайся по-мужски, без сюсюканья.\n\n"
        f"ФАКТЫ О ПОЛЬЗОВАТЕЛЕ:\n{core_text}\n\n"
        "ТВОИ СТАРЫЕ ЗАМЕТКИ "
        "(используй, если они в тему):\n"
        f"{notes_text}\n\n"
        "ПРАВИЛА:\n"
        "1. Отвечай строго на русском.\n"
        "2. Не выдумывай данные. Если не знаешь — честно "
        "скажи «не знаю» и задай уточняющий вопрос.\n"
        "3. В начале ответа напиши [CONFIDENCE: число], "
        "где число — целое от 0 до 100.\n"
        "   Для фактов о пользователе опирайся на его текущее "
        "сообщение и сохранённую память. Если нужных данных "
        "нет — уверенность не выше 50.\n"
        "   Для общих вопросов оценивай уверенность без "
        "привязки к наличию заметок о пользователе.\n"
        "4. Если уверенность ниже 70 — вместо ответа задай "
        "уточняющий вопрос.\n"
        "5. В конце ответа напиши [NOTE: краткая суть обсуждения]. "
        "Не записывай предположения как подтверждённые факты.\n"
        "6. Если пользователь явно сообщил важный факт о себе, "
        "добавь [CORE: факт]. Не выдумывай факты о пользователе.\n"
        "7. Служебные метки пиши строго в квадратных скобках. "
        "Внутри значений NOTE и CORE не используй квадратные "
        "скобки. Каждую метку обязательно закрывай символом ]."
    )

    messages_payload = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": final_user_prompt
        }
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages_payload,
        "temperature": 0.7,
        "max_tokens": 2048,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }

    # 4. Запрос к модели и статус «печатает».
    stop_typing = threading.Event()

    typing_thread = threading.Thread(
        target=keep_typing,
        args=(message.chat.id, stop_typing),
        daemon=True
    )
    typing_thread.start()

    try:
        response = requests.post(
            LLM_URL,
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        result = response.json()

        if not isinstance(result, dict):
            raise ValueError("Сервер модели вернул не JSON-объект.")

        choices = result.get("choices")

        if not isinstance(choices, list) or not choices:
            raise ValueError(
                "В ответе сервера модели отсутствует список choices."
            )

        choice = choices[0]

        if not isinstance(choice, dict):
            raise ValueError("Некорректный элемент choices.")

        model_message = choice.get("message")

        if not isinstance(model_message, dict):
            raise ValueError("В ответе модели отсутствует message.")

        raw_reply = model_message.get("content")

        if not isinstance(raw_reply, str) or not raw_reply.strip():
            raise ValueError("Модель вернула пустой текст ответа.")

    finally:
        stop_typing.set()
        typing_thread.join(timeout=1)

    # 5. Разбираем ответ.
    (
        confidence,
        core_memories,
        note_memories,
        clean_reply
    ) = parse_model_reply(raw_reply)

    if not clean_reply:
        clean_reply = (
            "Не получилось сформировать ответ. "
            "Повтори вопрос, пожалуйста."
        )
        confidence = 0

    # 6. Обновляем память.
    with state_lock:
        memory_changed = False

        for memory in core_memories:
            if memory not in profile["core"]:
                profile["core"].append(memory)
                memory_changed = True

        for memory in note_memories:
            if memory not in profile["notes"]:
                profile["notes"].append(memory)
                memory_changed = True

        if len(profile["notes"]) > 200:
            profile["notes"] = profile["notes"][-200:]
            memory_changed = True

        if memory_changed:
            save_json(PROFILE_FILE, user_profiles)

    # 7. Формируем ответ пользователю.
    if confidence < 70:
        pending_clarifications[chat_id] = (
            final_user_prompt,
            clean_reply
        )

        final_text = (
            f"Нужно уточнение (уверенность {confidence}):\n\n"
            f"{clean_reply}"
        )
    else:
        pending_clarifications.pop(chat_id, None)

        final_text = (
            f"{clean_reply}\n\n"
            f"Уверенность: {confidence}"
        )

    send_long_reply(message, final_text)


if __name__ == "__main__":
    print("Бот запущен в режиме Stateless.")
    bot.infinity_polling(
        timeout=60,
        long_polling_timeout=60
    )
