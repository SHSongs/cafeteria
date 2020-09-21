
import discord

client = discord.Client()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('!급식실 줄'):
                await message.channel.send(file=discord.File('./server/images/cafeteria.png'))

client.run('NzU3NDY4MjkzMDIzMDA2Nzkx.X2g1Ug.BvxaueB0ofAVapbYguURYPRRU20')
