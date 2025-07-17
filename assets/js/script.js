document.addEventListener('DOMContentLoaded', function() {
    // --- 页面切换逻辑 ---
    const navItems = document.querySelectorAll('.nav-item');
    const pages = document.querySelectorAll('.page');

    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();

            // 移除所有导航项的 active 类
            navItems.forEach(nav => nav.classList.remove('active'));
            // 为当前点击的导航项添加 active 类
            this.classList.add('active');

            const targetId = this.getAttribute('data-target');
            
            // 隐藏所有页面
            pages.forEach(page => page.classList.remove('active'));
            // 显示目标页面
            document.getElementById(targetId).classList.add('active');
        });
    });

    // --- AI 聊天机器人逻辑 ---
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    // 预设的AI知识库 (关键词 -> 回答)
    const aiKnowledgeBase = {
        '高血压': '高血压在高原地区很常见。主要是因为血管里的压力太高了。您需要注意少吃盐，多走动，保持心情愉快。如果医生开了药，一定要坚持吃。详细信息可以查看我们的【高原高血压管理指南】。',
        '头晕': '头晕的原因有很多，可能是高血压引起的，也可能是休息不好或者感冒了。如果经常头晕，建议您去医院检查一下，看看是哪种情况。可以先去【实用资源】板块查找离您最近的医院。',
        '吃饭': '健康的饮食很重要。要多吃蔬菜，主食（比如糌粑）和肉类要搭配好。尽量少吃外面卖的、太咸太油的食物。您可以参考我们的【藏式平衡膳食宝典】。',
        '吃什么': '健康的饮食很重要。要多吃蔬菜，主食（比如糌粑）和肉类要搭配好。尽量少吃外面卖的、太咸太油的食物。您可以参考我们的【藏式平衡膳食宝典】。',
        '睡不着': '睡不着可能是心里的“隆”不平顺了。睡前可以喝杯温牛奶，听一些舒缓的音乐，不要想太多烦心事。如果情况一直没有改善，可以看看我们的【身心安康之道】指南，或者咨询医生。',
        '失眠': '睡不着可能是心里的“隆”不平顺了。睡前可以喝杯温牛奶，听一些舒缓的音乐，不要想太多烦心事。如果情况一直没有改善，可以看看我们的【身心安康之道】指南，或者咨询医生。',
        '你好': '扎西德勒！有什么健康问题可以问我。',
        '谢谢': '不客气，愿您吉祥安康！'
    };

    function handleUserQuery() {
        const userText = userInput.value.trim();
        if (userText === '') return;

        // 在聊天框显示用户消息
        appendMessage(userText, 'user');
        userInput.value = '';

        // 模拟AI思考
        setTimeout(() => {
            const aiResponse = getAIResponse(userText);
            appendMessage(aiResponse, 'ai');
        }, 500);
    }

    function getAIResponse(userText) {
        // 查找关键词
        for (const keyword in aiKnowledgeBase) {
            if (userText.includes(keyword)) {
                return aiKnowledgeBase[keyword];
            }
        }
        // 默认回答
        return '抱歉，您的问题我暂时还不太明白。您可以换个方式问，比如“高血压怎么办？”或者“头晕是怎么回事？”。';
    }

    function appendMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        
        const paragraph = document.createElement('p');
        paragraph.textContent = text;
        
        messageDiv.appendChild(paragraph);
        chatBox.appendChild(messageDiv);
        
        // 滚动到底部
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    sendBtn.addEventListener('click', handleUserQuery);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            handleUserQuery();
        }
    });
});
