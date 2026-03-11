import json
from aiokafka import AIOKafkaProducer
import logging

logger = logging.getLogger(__name__)

# This will map to the Kafka broker address locally or cloud
KAFKA_BROKER = "localhost:29092"
TELEMETRY_TOPIC = "student_telemetry"

class KafkaProducerManager:
    producer: AIOKafkaProducer = None

    @classmethod
    async def start(cls):
        try:
            cls.producer = AIOKafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            await cls.producer.start()
            logger.info("Kafka Producer started successfully.")
        except Exception as e:
            logger.error(f"Failed to start Kafka Producer: {e}")

    @classmethod
    async def stop(cls):
        if cls.producer:
            await cls.producer.stop()
            logger.info("Kafka Producer stopped.")

    @classmethod
    async def send_event(cls, event: dict):
        if cls.producer is None:
            logger.warning(f"Kafka producer not initialized. Simulating send: {event}")
            return True
            
        try:
            await cls.producer.send_and_wait(TELEMETRY_TOPIC, event)
            return True
        except Exception as e:
            logger.error(f"Error sending Kafka message, simulating send: {event}. Error: {e}")
            return True
