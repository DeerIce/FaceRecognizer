import openai

openai.api_key = "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"

# model="使用的模型"
# prompt="要问的问题"
# max_tokens=返回结果字符的最大个数
response = openai.Completion.create(
    model="text-davinci-003",
    prompt="你能帮我写一份通过机器学习完成人脸识别的代码吗?",
    # temperature=0.7,
    max_tokens=900,
    # top_p=1,
    # frequency_penalty=0,
    # presence_penalty=0
)
# 打印结果
result = response.choices[0].text
print('result: ', result)
