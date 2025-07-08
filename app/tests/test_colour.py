# test_click_colors.py
import click

@click.command()
def hello():
    click.secho("This is green bold text!", fg="green", bold=True)
    click.secho("This is yellow text!", fg="yellow")
    click.echo("This is default text.")

if __name__ == "__main__":
    hello()
