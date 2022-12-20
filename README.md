# Install

```
$ conda env create -f conda_env.yaml
```

# How to use

1. Create `account.csv` with first line: `gameName,tagLine`.
   * If you want to connect Riot ID to Discord ID, create 'discord.csv with `gameName, tagLine, discordId`.
2. Add `discord_token.py` containing `DISCORD_TOKEN="YOUR_TOKEN"` variable.
3. Run `$ python run.py`.
   * Press `CTRL+C` to restart.
   * If you want to close the program, press `CTRL+C` again.


# Tested Environment
```
Windows 10, Python 3.10.8co
```