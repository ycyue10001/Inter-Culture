document.addEventListener('DOMContentLoaded', function() {
    // --- 页面切换逻辑 ---
    const navItems = document.querySelectorAll('.nav-item');
    const pages = document.querySelectorAll('.page');
    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            navItems.forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');
            const targetId = this.getAttribute('data-target');
            pages.forEach(page => page.classList.remove('active'));
            document.getElementById(targetId).classList.add('active');
        });
    });

    // --- AI 聊天机器人逻辑 ---
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const GEMINI_API_KEY = 'YOUR_API_KEY'; // **重要：请替换为您的API密钥**
    const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${GEMINI_API_KEY}`;

    async function getGeminiResponse(prompt) {
        const loadingMessage = appendMessage('正在思考中...', 'ai loading');
        const fullPrompt = `你是一个名为 "Amchi-AI" (智医) 的健康问答助手，专门为生活在高原地区的藏族用户服务。你的回答应遵循以下原则：1. **文化敏感性**：在可能的情况下，结合藏医药中“隆、赤巴、培根”三因平衡的理念来解释健康问题。2. **简洁易懂**：使用简单、清晰、尊重的语言，避免复杂的医学术语。3. **安全第一**：必须在所有回答的末尾加上一句：“请注意，我的回答仅供参考，不能替代专业医生的诊断。如果身体不适，请及时就医。”4. **知识范围**：你的知识应主要围绕高血压、糖尿病、饮食健康、心理平衡、包虫病预防等高原常见健康问题。如果问题超出范围，应礼貌地表示无法回答。现在，请回答用户的问题： "${prompt}"`;
        try {
            const response = await fetch(GEMINI_API_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ contents: [{ parts: [{ text: fullPrompt }] }] }) });
            chatBox.removeChild(loadingMessage);
            if (!response.ok) throw new Error(`API请求失败，状态码: ${response.status}`);
            const data = await response.json();
            return data.candidates.content.parts.text;
        } catch (error) {
            console.error('调用Gemini API时出错:', error);
            if (chatBox.contains(loadingMessage)) chatBox.removeChild(loadingMessage);
            return '抱歉，我的网络好像出了一点问题，暂时无法回答。请稍后再试。';
        }
    }

    async function handleUserQuery() {
        const userText = userInput.value.trim();
        if (userText === '') return;
        appendMessage(userText, 'user');
        userInput.value = '';
        const aiResponse = await getGeminiResponse(userText);
        appendMessage(aiResponse, 'ai');
    }

    function appendMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        const paragraph = document.createElement('p');
        paragraph.textContent = text;
        messageDiv.appendChild(paragraph);
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        return messageDiv;
    }

    sendBtn.addEventListener('click', handleUserQuery);
    userInput.addEventListener('keypress', function(e) { if (e.key === 'Enter') handleUserQuery(); });

    // --- 健康指南模态窗口逻辑 ---
    const modal = document.getElementById('guide-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    const closeBtn = document.querySelector('.close-btn');

    // 指南详细内容 (为了代码简洁，内容放在JS中)
    const guideDetails = {
        'guide-1': {
            title: '高原高血压管理指南',
            content: `<h4>什么是高血压？</h4><p>在藏医药的智慧中，身体的健康依赖于“隆”（风）、“赤巴”（胆）、“培根”（涎）三种能量的平衡。当“隆”的运行不畅或过盛时，可能会引起头晕、心慌等不适。现代医学把血管里血液的压力持续过高的情况称为“高血压”，这会增加心脏和脑血管疾病的风险。</p><h4>生活中的管理智慧：</h4><ul><li><strong>饮食调理——少一点盐，多一份健康：</strong> 传统饮食中的酥油茶和风干肉虽然美味，但含盐量较高，过多的盐分会给身体带来负担。<strong>建议：</strong> 尝试每天在茶里少放一撮盐，慢慢品味牛奶和茶本身的香醇。烹饪时，多用葱、姜、蒜等天然香料提味。</li><li><strong>规律运动——让“隆”顺畅起来：</strong> 每日转经、散步都是非常好的活动。保持适度运动，有助于气血流通，平衡“隆”的能量。<strong>建议：</strong> 争取每天能有半小时左右的快走，让心跳微微加快，身体微微出汗。</li><li><strong>遵从医嘱——现代方法的帮助：</strong> 如果医生为您开了降压药，请把它看作是帮助身体恢复平衡的有力工具。不要因为感觉好些就随意停药，务必按照医生的指导坚持服用。</li><li><strong>心态平和——安抚内在的风：</strong> 焦虑和压力也会扰乱“隆”的平衡。<strong>建议：</strong> 感到心烦时，可以静坐几分钟，专注于自己的呼吸，感受气息的一进一出，这能帮助内心恢复平静。</li></ul>`
        },
        'guide-2': {
            title: '藏式平衡膳食宝典',
            content: `<h4>藏式“平衡餐盘”：</h4><p>我们的传统食物是高原赐予的宝贵财富。用现代营养的眼光来看，我们可以这样搭配：</p><ul><li><strong>一半蔬菜：</strong> 尽可能让蔬菜占到餐盘的一半，比如萝卜、土豆、白菜、青笋等。</li><li><strong>四分之一主食：</strong> 糌粑是极好的主食，富含纤维，能量持久。</li><li><strong>四分之一蛋白：</strong> 牦牛肉、羊肉和酸奶、奶渣提供了优质的蛋白质。烹饪时多采用炖、煮的方式。</li></ul><h4>读懂食品包装的秘密：</h4><p>研究发现，很多人不了解如何看包装食品的说明。购买时，请花一点时间看看包装背面：</p><ul><li><strong>看配料表：</strong> 排在越前面的成分，含量越高。如果“糖”、“盐”、“油”（或“脂肪”）排在前几位，就要少吃。</li><li><strong>看营养成分表：</strong> 关注“钠”（代表盐）、“脂肪”和“糖”的含量。这些数值越低越好。</li></ul><h4>特别提醒：预防包虫病</h4><p>这是我区常见的、可以预防的疾病。请一定记住以下几点：</p><ol><li><strong>管好家犬：</strong> 定期给家里的狗喂驱虫药，这是最重要的一步。</li><li><strong>喝开水，勤洗手：</strong> 不喝生水，饭前便后、处理完生肉、接触过犬只后，要用肥皂彻底洗手。</li><li><strong>食物要洗净煮熟：</strong> 蔬菜水果要洗干净，肉类一定要煮熟再吃。</li></ol>`
        },
        //... 为其他10份指南添加类似的内容...
        'guide-6': {
            title: '高原关节健康与养护',
            content: `<h4>为何高原地区关节易出问题？</h4><p>高原气候寒冷、潮湿、气压低，这些因素都可能加重关节的负担，导致疼痛、僵硬等问题，中医和藏医都认为“寒湿”是关节问题的常见原因。</p><h4>日常养护建议：</h4><ul><li><strong>保暖是第一要务：</strong> 尤其要注意膝盖、腰部等关键部位的保暖。天气转凉时，及时添加衣物，可以佩戴护膝。</li><li><strong>合理运动，避免劳损：</strong> 长期不活动会让关节僵硬，但过度或不当的运动会加重磨损。<strong>建议：</strong> 散步、打太极拳等舒缓的运动非常适合。避免长时间下蹲或搬运重物。</li><li><strong>食疗辅助：</strong> 可以适量食用一些有温通经络作用的食物，如生姜、花椒等。避免过多食用生冷寒凉的食物。</li></ul>`
        },
        'guide-10': {
            title: '高原眼部健康与保护',
            content: `<h4>高原阳光的“双刃剑”：</h4><p>高原的阳光虽然明媚，但强烈的紫外线是眼睛的“隐形杀手”，长期暴露会大大增加患白内障、翼状胬肉等眼病的风险。</p><h4>如何保护我们的“心灵之窗”：</h4><ul><li><strong>佩戴合格的太阳镜：</strong> 这不是为了时尚，而是必需品。选择能明确标识阻挡UVA和UVB的太阳镜，最好能包裹住眼周，减少侧面进入的光线。</li><li><strong>宽檐帽也是好帮手：</strong> 戴一顶宽檐帽可以进一步遮挡来自上方的紫外线。</li><li><strong>合理用眼，避免疲劳：</strong> 即使在室内，长时间看手机、看电视也会让眼睛疲劳。<strong>建议：</strong> 每隔半小时左右，向远处眺望一会儿，让眼睛得到休息。</li><li><strong>补充营养：</strong> 多吃富含维生素A和叶黄素的食物，如胡萝卜、玉米、深绿色蔬菜等，对眼睛有好处。</li></ul>`
        }
    };

    document.querySelectorAll('.guide-btn').forEach(button => {
        button.addEventListener('click', function() {
            const guideId = this.parentElement.getAttribute('data-guide');
            const details = guideDetails[guideId];
            if (details) {
                modalTitle.textContent = details.title;
                modalBody.innerHTML = details.content;
                modal.style.display = 'block';
            } else { // 为新增的、内容未完全定义的指南提供默认文本
                 modalTitle.textContent = this.parentElement.querySelector('h4').textContent;
                 modalBody.innerHTML = '<p>详细内容正在撰写中，敬请期待...</p>';
                 modal.style.display = 'block';
            }
        });
    });

    closeBtn.onclick = function() { modal.style.display = 'none'; }
    window.onclick = function(event) { if (event.target == modal) { modal.style.display = 'none'; } }
    window.onkeydown = function(event) { if (event.key === 'Escape') { modal.style.display = 'none'; } }
});
