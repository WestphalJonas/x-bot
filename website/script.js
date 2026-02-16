(function () {
    const menuToggle = document.getElementById("menu-toggle");
    const nav = document.getElementById("site-nav");
    const navLinks = Array.from(document.querySelectorAll("#site-nav a[href^='#']"));
    const sections = navLinks
        .map((link) => document.querySelector(link.getAttribute("href")))
        .filter(Boolean);

    if (menuToggle && nav) {
        menuToggle.addEventListener("click", () => {
            nav.classList.toggle("open");
        });

        navLinks.forEach((link) => {
            link.addEventListener("click", () => nav.classList.remove("open"));
        });
    }

    const revealItems = document.querySelectorAll(".reveal");
    const revealObserver = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("visible");
                    revealObserver.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.12 }
    );

    revealItems.forEach((item) => revealObserver.observe(item));

    const sectionObserver = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) {
                    return;
                }

                const currentId = `#${entry.target.id}`;
                navLinks.forEach((link) => {
                    const active = link.getAttribute("href") === currentId;
                    link.classList.toggle("active", active);
                });
            });
        },
        {
            threshold: 0.45,
            rootMargin: "-10% 0px -40% 0px",
        }
    );

    sections.forEach((section) => sectionObserver.observe(section));
})();
