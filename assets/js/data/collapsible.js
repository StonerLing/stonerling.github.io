document.addEventListener("DOMContentLoaded", function () {
    let coll = document.getElementsByClassName("collapsible-container");
    let maxLines = 5; // 设置折叠显示的行数 Maximum number of lines to display without collapsing
    let defaultOpenLines = 10; // 默认展开的最大行数

    for (let i = 0; i < coll.length; i++) {
      let trigger = coll[i].querySelector('.collapsible-trigger');
      let content = coll[i].querySelector('.collapsible-content');
      
      // 检查 trigger 和 content 是否存在
      if (!trigger || !content) continue;

      let codeLines = (content.textContent.split('\n').length - 1) / 2; // 计算实际的代码行数
      //console.log(codeLines);

      if (codeLines <= defaultOpenLines) {
        trigger.innerHTML = "收起";
        trigger.style.display = 'none'; // 隐藏触发器
        content.style.maxHeight = content.scrollHeight + "px";
      } else {
        trigger.addEventListener("click", function () { 
          // 切换按钮上的文字
          if (this.innerHTML.includes("展开")) {
            this.innerHTML = "收起";
          } else {
            this.innerHTML = "展开";
          }
          this.classList.toggle("active");
          if (content.style.maxHeight) {
            content.style.maxHeight = null;
            // 滚动页面到 trigger 元素的位置
            content.scrollIntoView({
              behavior: "smooth",
              block: "center"
            });
          } else {
            content.style.maxHeight = content.scrollHeight + "px";
          }
        });
      }
    }
});