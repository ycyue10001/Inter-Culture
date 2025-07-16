/**
 * "扎西德勒健康" 网站交互脚本
 * 本脚本实现简单的客户端交互，以增强用户体验，但保持轻量，不依赖任何库。
 * 交互设计遵循“简化”原则，以适应低数字素养用户。
 * 深层结构适配 (Deep Structure Adaptation)
 * Ref: [6, 12]
 */

document.addEventListener('DOMContentLoaded', function() {

    // --- 平滑滚动导航 ---
    // 为导航链接添加点击事件监听器，实现平滑滚动到页面锚点。
    const navLinks = document.querySelectorAll('header nav a');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // --- 健康主题详情展示 ---
    // 为所有“查看详情”按钮添加点击事件，动态显示对应的隐藏内容。
    const detailButtons = document.querySelectorAll('.details-btn');
    const detailsContainer = document.getElementById('details-container');
    let currentOpenContent = null;

    detailButtons.forEach(button => {
        button.addEventListener('click', function() {
            const contentId = this.getAttribute('data-content');
            const contentElement = document.getElementById(contentId);

            if (contentElement) {
                // 如果点击的是已经打开的按钮，则关闭详情
                if (currentOpenContent === contentId) {
                    detailsContainer.style.display = 'none';
                    currentOpenContent = null;
                } else {
                    // 否则，显示新的详情内容
                    detailsContainer.innerHTML = contentElement.innerHTML;
                    detailsContainer.style.display = 'block';
                    currentOpenContent = contentId;
                    
                    // 滚动到详情容器，确保用户能看到内容
                    detailsContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }
        });
    });

});
