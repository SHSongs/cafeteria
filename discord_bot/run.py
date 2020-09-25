
import discord

client = discord.Client()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    file = open("C:/cafeteria/people-counter/cnt.txt", 'r')
    cnt = int(file.read())
    file.close()

    if message.content.startswith('!급식실 줄'):
                await message.channel.send(file=discord.File('C:/cafeteria/server/images/img'+str(cnt)+'.jpg'))

client.run('')
