import DefaultTheme from "vitepress/theme";
import type { Theme } from "vitepress";
import { h, onBeforeUnmount, onMounted } from "vue";
import LandingPage from "./components/LandingPage.vue";
import "./custom.css";

const BookLayout = {
  name: "BookLayout",
  setup() {
    const restoreSearchFocus = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || !document.querySelector(".VPLocalSearchBox")) {
        return;
      }

      window.setTimeout(() => {
        if (!document.querySelector(".VPLocalSearchBox")) {
          document
            .querySelector<HTMLButtonElement>(".VPNavBarSearch button")
            ?.focus();
        }
      });
    };

    onMounted(() => {
      document.addEventListener("keydown", restoreSearchFocus, true);
    });
    onBeforeUnmount(() => {
      document.removeEventListener("keydown", restoreSearchFocus, true);
    });

    return () => h(DefaultTheme.Layout);
  },
};

export default {
  extends: DefaultTheme,
  Layout: BookLayout,
  enhanceApp({ app }) {
    app.component("LandingPage", LandingPage);
  },
} satisfies Theme;
