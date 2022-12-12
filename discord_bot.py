from opgg_crawler import get_stats
import discord

class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        print(f'Message from {message.author}: {message.content}')
        if message.content.startswith('!스탯'):
            pending_message = await message.channel.send("10초정도 기다리세용")
            stats = await get_stats()
            await pending_message.delete()
            await message.channel.send(content=stats)

intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)
client.run('MTA1MTg0MTI0ODI3MDQ4NzY2Mw.GZH04H.DnOmJoHYEVTxndrVkyFMkpdQ4v9fgBzTZ1aoQc')