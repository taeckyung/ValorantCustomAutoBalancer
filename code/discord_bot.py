from discord_token import DISCORD_TOKEN
import bot_api
import discord


class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message: discord.Message):
        print(f'Message from {message.author}: {message.content}')
        content = message.content
        if content.startswith('!내전'):
            content = content.replace('!내전','').strip()
            if content.startswith('스탯'):
                result, fname = await bot_api.get_stats()
                if fname is None:
                    await message.channel.send(content=result)
                else:
                    await message.channel.send(content=result, file=discord.File(fname))
            elif content.startswith('멤버'):
                content = content.replace('멤버','').strip()
                if content.startswith('추가'):
                    content = content.replace('추가','').strip()
                    result = await bot_api.add_member(content)
                    await message.channel.send(content=result)
                elif content.startswith('삭제') or content.startswith('제거'):
                    content = content.replace('삭제','').replace('제거','').strip()
                    result = await bot_api.remove_member(content)
                    await message.channel.send(content=result)
                else:
                    result = await bot_api.get_member()
                    await message.channel.send(content=result)
            elif content.startswith('업데이트'):
                pending_message = await message.channel.send("업데이트중... 기다리세용")
                result = await bot_api.update()
                await pending_message.delete()
                await message.channel.send(content=result)
            elif content.startswith('자동밸런스') or content.startswith('자밸'):
                result = await bot_api.auto_balance()
                result = await bot_api.riotID_to_discord(result)
                await message.channel.send(content=result)
            elif content.startswith('맵'):
                result = await bot_api.random_map()
                await message.channel.send(content=result)
            else:
                await message.channel.send(content="사용법: `!내전 멤버` `!내전 업데이트` `!내전 맵` `!내전 멤버 추가 [이름]` `!내전 멤버 삭제 [이름]` `!내전 자동밸런스` `!내전 스탯`")
        elif content.startswith('!냥'):
            content = content.replace('!냥','').strip()
            await message.delete()
            await message.channel.send(content=content)


intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)
client.run(DISCORD_TOKEN)

        