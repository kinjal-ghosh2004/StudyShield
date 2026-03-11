import asyncio
import asyncpg

async def main():
    try:
        conn = await asyncpg.connect('postgresql://ai_user:ai_password@localhost:5433/dropout_prevention')
        print("Success")
        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(main())
