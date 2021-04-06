import nox


@nox.session
def lint(session):
    session.install("black")
    session.run("black", "--check", "-l", "99", ".")


@nox.session
def typing(session):
    session.install("mypy")
    print("\033[1m------> TYPING: Ignore errors related to variable reuse <------\033[0m")
    session.run("mypy", "--namespace-packages", "-p", "src")
