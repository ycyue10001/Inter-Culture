// 标签页切换功能
function openTab(evt, tabName) {
    var i, tabcontent, tablinks;

    // 获取所有标签内容并隐藏
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
        tabcontent[i].classList.remove("active");
    }

    // 获取所有标签链接并移除 "active" 类
    tablinks = document.getElementsByClassName("tab-link");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    // 显示当前点击的标签内容，并为其链接添加 "active" 类
    document.getElementById(tabName).style.display = "block";
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.className += " active";
}

// 默认打开第一个标签页
document.addEventListener("DOMContentLoaded", function() {
    document.querySelector('.tab-link').click();
});
