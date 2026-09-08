from langserve import RemoteRunnable

if __name__ == '__main__':
    client = RemoteRunnable("http://localhost:8000/langchainServer")
    print(client.invoke({"language": "英文", "input": "我喜欢编程！"}))