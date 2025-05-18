import json
import logging
import traceback
import asyncio

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from app.services.chat_service import chat_stream

from app.services.session_config import SessionConfig

LOG = logging.getLogger("kafka-handler")

KAFKA_BOOTSTRAP      = "localhost:9092"
REQUEST_TOPIC        = "chat-requests"
RESPONSE_TOPIC       = "chat-responses"
GROUP_ID             = "fastapi-consumer-group"
SESSION_TIMEOUT_MS   = 300_000   # 5분
MAX_POLL_INTERVAL_MS = 300_000   # 5분

consumer: AIOKafkaConsumer
producer: AIOKafkaProducer

async def init_kafka():
    global consumer, producer
    consumer = AIOKafkaConsumer(
        REQUEST_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=GROUP_ID,
        session_timeout_ms=SESSION_TIMEOUT_MS,
        max_poll_interval_ms=MAX_POLL_INTERVAL_MS,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
    )
    await consumer.start()
    await producer.start()
    LOG.info("Kafka initialized")

async def shutdown_kafka():
    await consumer.stop()
    await producer.stop()
    LOG.info("Kafka shutdown complete")

async def consume_loop():
    LOG.info("Starting consume loop")
    async for msg in consumer:
        asyncio.create_task(handle_message(msg))

async def handle_image(data: dict[str, any]):
    b64_img = data.get('content')
    # 이미지 검증

    # 일반 이미지면
    # if 
    # b64_str = base64.b64encode(raw_bytes).decode("utf-8")

    # 메시지 형식 만들기
    # data_uri = f"data:{file.content_type};base64,{b64_str}"
    # human_msg = HumanMessage(
    #     content=[
    #         {"type": "image_url", "image_url": data_uri},
    #         {"type": "text",      "text": "What's this? Provide a description in korean without leading or trailing text or markdown syntax."}
    #     ]
    # )

    # 답변 요청
    # NOTE: singleton 공유 가능한 모델 정의 필요
    # llm = ChatOllama(
    #     model="gemma3:4b",              
    #     temperature=0.0,
    # )

    # 답변을 session_id를 통해 수동으로 업데이트
    # ai_msg = llm.invoke([human_msg])

async def handle_message(msg):
    LOG.debug(msg)
    try:
        data = msg.value
        # 이미지 일 때
        if data.get("type") != "TEXT":
            await consumer.commit()
            await handle_image()
            return

        user = data["from"]

        loop   = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, to_response_type, data)

        await producer.send_and_wait(
            RESPONSE_TOPIC,
            key=user,
            value=result
        )
        await consumer.commit()
    except Exception:
        LOG.exception("Error processing message")

def to_response_type(msg: dict) -> dict:
    content : str = ""
    try:
        session_config = SessionConfig(msg.get("from"), msg.get('to'))
        prompt = str(msg.get('content'))
        if len(prompt) == 0:
            raise Exception
        
        # NOTE 문장 단위로 Produce 로직 필요
        for chunk in chat_stream(prompt, session_config):
            content += chunk

    except Exception as e:
        LOG.error(f"메시지 처리간 오류가 발생했습니다. from:{msg.get('from')} to:{msg.get('to')} {traceback.format_exc()} {e}")
        
    finally:
        # NOTE 메시지 타입 지정 필요 (ERROR, NORMAL ...)
        return {
            "chatRoomId":msg.get('chatRoomId'),
            "from":      msg.get('to'),
            "to":        msg.get('from'),
            "content":   content,
            "type":      "TEXT",
            "mcpEnabled": False,
        }

