import subprocess

def serve():
    subprocess.run(["mkdocs", "serve"], check=True)

def build():
    subprocess.run(["mkdocs", "build"], check=True)
