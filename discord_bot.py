from discord_token import DISCORD_TOKEN
import opgg_crawler
import traceback
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
                pending_message = await message.channel.send("10초정도 기다리세용")
                total_match_cnt, latest_match_time, _ = await opgg_crawler.get_stats()
                string = f"Total Matches: {total_match_cnt} \t|\t Latest Match: {latest_match_time.isoformat()} \t|\t rating := sqrt(win) * score\n"
                await pending_message.delete()
                await message.channel.send(content=string, file=discord.File("table.png"))
            elif content.startswith('멤버'):
                content = content.replace('멤버','').strip()
                if content.startswith('추가'):
                    content = content.replace('추가','').strip()
                    result = await opgg_crawler.add_member(content)
                    await message.channel.send(content=result)
                elif content.startswith('삭제') or content.startswith('제거'):
                    content = content.replace('삭제','').replace('제거','').strip()
                    result = await opgg_crawler.remove_member(content)
                    await message.channel.send(content=result)
                else:
                    result = await opgg_crawler.get_member()
                    await message.channel.send(content=result)
            elif content.startswith('자동밸런스') or content.startswith('자밸'):
                pending_message = await message.channel.send("10초정도 기다리세용")
                result = await opgg_crawler.auto_balance()
                result = await opgg_crawler.riotID_to_discord(result)
                await pending_message.delete()
                await message.channel.send(content=result)
            else:
                await message.channel.send(content="사용법: `!내전 멤버` `!내전 멤버 추가 [이름]` `!내전 멤버 삭제 [이름]` `!내전 자동밸런스` `!내전 스탯`")
        elif content.startswith('!') and message.author.id in [329146794128834570, 331845856477052928, 391487027855622145]:
            content = content.replace('!','').strip()
            await message.channel.send(content=content)


intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)
client.run(DISCORD_TOKEN)

        