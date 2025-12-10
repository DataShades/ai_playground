import click

from ai_pg.commands import get_commands


@click.group()
def main():
    pass


for command in get_commands():
    main.add_command(command)

if __name__ == "__main__":
    main()
