
![ComfyUI DeepSeek JanusPro 封面图](https://github.com/user-attachments/assets/9f8ef93e-067c-42af-9eaf-d0f0fbd427c4)


# ComfyUI-DeepSeek-JanusPro（项目说明+细节还在完善中，代码已经可以使用）

<img width="1101" alt="截屏2025-01-30 02 06 33" src="https://github.com/user-attachments/assets/defb2260-25af-4c5c-a534-c948a055b456" />

## 由 DeepSeek R1 成功独立完成代码（指：我未写、我未了解原项目代码、我未检查代码）

DeepSeek R1 自己给自己的 JanusPro 成功写好 ComfyUI 插件（我没写一行！

关键点：之前是 LLM 辅助我写插件，我还得了解代码本身，现在几乎无脑给 R1 就能直接交付了

无需微调直接就成，无需人看代码/写代码，细节准确度高，预计交互次数理想状态下可以控制在 3-5 次以内（标准是直接就能在 ComfyUI 成功运行），体感比 O1 的细节/准确度更好（还需进一步验证


## 具体过程如下

1）我的角色：信息传递员+判断者，我没看 JanusPro 代码，直接都丢给 R1 处理

2）给 R1 的样本学习：我自己写的 Emu3 插件的完整代码（两者架构不同

3）把 JanusPro 的官方 demo 代码丢给 R1 

4）R1 先将其分为3个核心节点，然后写出了完整代码，并对其做了优化和兼容性考虑（增强，还给出了使用方式和建议参数范围

5）运行之后遇到第一次报错（1个，我提出要求之后 R1 完成修改

6）运行之后遇到第二次报错（2个，成功解决，但是由于报错之后未运行第二项功能的节点，所以我提出同样也需要修改，R1 完成修改，但是漏掉了部分关键格式

7）补充完整遗漏，第一部分功能已经实现可以正常运行

8）第二部分功能 R1 做了过度思考和复杂化，导致偏离原代码，我在发现此现象后，向其提出是否已经偏离原代码，请检查，R1 回顾之前报错并纠正偏离，第二部分也成功实现并运行，运行结果如下图


## 部分思考过程截图

<img width="816" alt="截屏2025-01-30 02 50 57" src="https://github.com/user-attachments/assets/83f6196f-6469-4b2e-84af-7892aad437ce" />

<img width="424" alt="截屏2025-01-30 02 51 47" src="https://github.com/user-attachments/assets/c998dfce-ce73-404b-ad2b-63185e424c2a" />


## 使用示例：

<img width="1110" alt="截屏2025-01-30 02 13 08" src="https://github.com/user-attachments/assets/e9596c12-7cb8-4fb4-8369-ba608a8b5205" />


## 更新日志

- 20250221 新增封面图，并将会合并到新的大项目中：[DeepSeek|All-In-One|ComfyUI](https://github.com/ZHO-ZHO-ZHO/ComfyUI-DeepSeek-All-In-One)

- 20250130（大年初二）

  V1.0 由 DeepSeek R1 成功独立完成代码（指：我未写、我未了解原项目代码、我未检查代码）

  创建项目
  

## Stars 

[![Star History Chart](https://api.star-history.com/svg?repos=ZHO-ZHO-ZHO/ComfyUI-DeepSeek-JanusPro&type=Date)](https://star-history.com/#ZHO-ZHO-ZHO/ComfyUI-DeepSeek-JanusPro&Date)


## 关于我 | About me

📬 **联系我**：
- 邮箱：zhozho3965@gmail.com
- QQ 群：839821928

🔗 **社交媒体**：
- 个人页：[-Zho-](https://jike.city/zho)
- Bilibili：[我的B站主页](https://space.bilibili.com/484366804)
- X（Twitter）：[我的Twitter](https://twitter.com/ZHO_ZHO_ZHO)
- 小红书：[我的小红书主页](https://www.xiaohongshu.com/user/profile/63f11530000000001001e0c8?xhsshare=CopyLink&appuid=63f11530000000001001e0c8&apptime=1690528872)

💡 **支持我**：
- B站：[B站充电](https://space.bilibili.com/484366804)
- 爱发电：[为我充电](https://afdian.com/a/ZHOZHO)


## Credits

[Janus](https://github.com/deepseek-ai/Janus/tree/main)
