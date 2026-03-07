import logging
import json
import asyncio
from aiokafka import AIOKafkaConsumer
from influxdb_client import Point
from agentic_system.backend.db.session import get_influx_client
from agentic_system.backend.core.config import settings

logger = logging.getLogger(__name__)

KAFKA_BROKER = "localhost:29092"
TELEMETRY_TOPIC = "student_telemetry"

class KafkaConsumerWorker:
    consumer: AIOKafkaConsumer = None
    _task: asyncio.Task = None

    @classmethod
    async def start(cls):
        try:
            cls.consumer = AIOKafkaConsumer(
                TELEMETRY_TOPIC,
                bootstrap_servers=KAFKA_BROKER,
                group_id="agentic_telemetry_group",
                auto_offset_reset="earliest",
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            await cls.consumer.start()
            logger.info("Kafka Consumer started successfully.")
            
            # Start the background loop
            cls._task = asyncio.create_task(cls.consume_loop())
            
        except Exception as e:
            logger.error(f"Failed to start Kafka Consumer: {e}")

    @classmethod
    async def stop(cls):
        if cls._task:
            cls._task.cancel()
            try:
                await cls._task
            except asyncio.CancelledError:
                pass
                
        if cls.consumer:
            await cls.consumer.stop()
            logger.info("Kafka Consumer stopped.")

    @classmethod
    async def consume_loop(cls):
        """
        Continuously polls Kafka for new telemetry events, processes them into
        InfluxDB points, and writes them asynchronously.
        """
        influx_client = get_influx_client()
        write_api = influx_client.write_api()
        
        logger.info("Listening for messages on Kafka topic...")
        
        try:
            async for msg in cls.consumer:
                event = msg.value
                student_id = event.get("student_id")
                event_type = event.get("event_type")
                payload = event.get("payload", {})
                
                # Transform raw event into an InfluxDB Timeseries Point
                point = (
                    Point("student_interaction")
                    .tag("student_id", student_id)
                    .tag("course_id", event.get("course_id", "Unknown"))
                    .tag("event_type", event_type)
                )
                
                # Append payload fields based on event type
                for k, v in payload.items():
                    if isinstance(v, (int, float)):
                        point.field(k, float(v))
                    elif isinstance(v, str):
                        point.field(k, v)
                
                # Write to InfluxDB immediately (In production, use batching)
                await write_api.write(bucket=settings.INFLUXDB_BUCKET, org=settings.INFLUXDB_ORG, record=point)
                
                logger.debug(f"Processed & Written to Influx: {student_id} | {event_type}")
                
        except asyncio.CancelledError:
            logger.info("Consumer loop cancelled.")
        except Exception as e:
            logger.error(f"Error in Consumer loop: {e}")
