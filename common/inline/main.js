/*
* 切换主题皮肤
* mode: 'light' | 'dark' | 'auto'
* */
function toggleTheme(mode = 'auto') {
    if (mode === 'dark') {
        //暗黑主题
        $('html')
            .addClass('dark')
            .removeClass('personal');
        //标记暗黑模式
        localStorage.setItem('theme-mode', 'dark');
        //改变图标和文字
        $(function () {
            $('.read-mode i')
                .removeClass("icon-baitian-qing icon-shezhi1")
                .addClass("icon-yueliang")
                .show();
            $('.read-mode .theme-mode-text').hide();
        });
    } else if (mode === 'light') {
        //白天主题
        $('html')
            .removeClass('dark')
            .addClass('personal');
        //标记白天模式
        localStorage.setItem('theme-mode', 'light');
        //改变图标和文字
        $(function () {
            $('.read-mode i')
                .removeClass("icon-yueliang icon-shezhi1")
                .addClass("icon-baitian-qing")
                .show();
            $('.read-mode .theme-mode-text').hide();
        });
    } else {
        //自动跟随系统
        localStorage.setItem('theme-mode', 'auto');
        //改变图标和文字
        $(function () {
            $('.read-mode i')
                .removeClass("icon-yueliang icon-baitian-qing icon-shezhi1")
                .hide();
            $('.read-mode .theme-mode-text').text('Auto').show();
        });
        //应用系统主题
        applySystemTheme();
    }
}

/*
* 应用系统主题
* */
function applySystemTheme() {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        //系统是暗色模式
        $('html')
            .addClass('dark')
            .removeClass('personal');
    } else {
        //系统是亮色模式
        $('html')
            .removeClass('dark')
            .addClass('personal');
    }
}

/*
* 监听系统主题变化
* */
if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
        const themeMode = localStorage.getItem('theme-mode');
        if (themeMode === 'auto') {
            applySystemTheme();
        }
    });
let systemThemeChangeListener = function(e) {
    const themeMode = localStorage.getItem('theme-mode');
    if (themeMode === 'auto') {
        applySystemTheme();
    }
};
if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', systemThemeChangeListener);
}


/*
* 动态rem
* */
let l = () => {
    let r = document.documentElement, o = r.offsetWidth / 100;
    o < 17 && (o = 17), r.style.fontSize = o + "px", window.rem = o
};
window.onresize = l;
l();

/*同步主题*/
let theme = localStorage.getItem('theme-color');
if (!!theme) {
    $('html').addClass(theme)
}

/*同步主题模式 */
let themeMode = localStorage.getItem('theme-mode');

// 兼容旧版本的 'night' 存储方式
let oldNight = localStorage.getItem('night');
if (!!oldNight && !themeMode) {
    themeMode = 'dark';
    localStorage.setItem('theme-mode', 'dark');
    localStorage.removeItem('night');
}

// 如果没有设置过模式，默认为自动跟随系统
if (!themeMode) {
    themeMode = 'auto';
    localStorage.setItem('theme-mode', 'auto');
}

/*
* 初始化主题模式
* */
toggleTheme(themeMode);

/*
* 获取元素在网页的实际top
* */
$.fn.getTop = function () {
    let position = this.position();
    /*
    * 为0代表有很多offsetTop要计算
    * */
    if (position.top !== 0) {
        return position.top;
    } else {
        let html = $('html').get(0);
        return this.get(0).getBoundingClientRect().top + html.scrollTop;
    }
}


/*jq内存清理函数*/
$.fn.removeWithLeakage = function () {
    this.each(function (i, e) {
        $("*", e).add([e]).each(function () {
            $.event.remove(this);
            $.removeData(this);
        });
        if (e.parentNode)
            e.parentNode.removeChild(e);
    });
};
