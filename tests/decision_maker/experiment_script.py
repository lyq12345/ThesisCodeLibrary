import subprocess

# 循环遍历1到20的数
for i in range(1, 21):
    # 构建要执行的命令
    command = f"python deployment_test.py 50 {i}"

    # 执行命令并捕获输出
    output = subprocess.check_output(command, shell=True, text=True)

    # 将输出写入文件
    with open(f"results/output_{i}.txt", "w") as file:
        file.write(output)

    print(f"执行完成：{i}")