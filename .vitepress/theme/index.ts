import DefaultTheme from "vitepress/theme";
import type { Theme } from "vitepress";
import { h, onBeforeUnmount, onMounted } from "vue";
import ChapterOpener from "./components/ChapterOpener.vue";
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

    /**
     * A wide diagram is scaled down to the text column, which can leave its labels too
     * small to read. Clicking one opens it in a dialog at up to the window's width.
     */
    let figureDialog: HTMLDialogElement | null = null;
    const enlargeFigure = (event: MouseEvent) => {
      const figure = (event.target as Element | null)?.closest?.(".vp-doc .mermaid");
      const source = figure?.querySelector("svg");
      if (!source || window.getSelection()?.toString()) {
        return;
      }

      if (!figureDialog) {
        figureDialog = document.createElement("dialog");
        figureDialog.className = "sdp-figure";
        figureDialog.setAttribute("aria-label", "Enlarged diagram");
        figureDialog.addEventListener("click", () => figureDialog?.close());
        figureDialog.addEventListener("close", () => figureDialog?.replaceChildren());
        document.body.append(figureDialog);
      }

      const natural = source.viewBox.baseVal.width || source.getBoundingClientRect().width;
      const width = Math.min(window.innerWidth * 0.94 - 56, natural * 2.2);
      const copy = source.cloneNode(true) as SVGSVGElement;
      copy.removeAttribute("height");
      copy.style.cssText = `display:block;width:${Math.round(width)}px;max-width:none;height:auto`;

      const sheet = document.createElement("div");
      sheet.className = "sdp-figure-sheet";
      sheet.append(copy);
      figureDialog.replaceChildren(sheet);
      figureDialog.showModal();
    };

    onMounted(() => {
      document.addEventListener("keydown", restoreSearchFocus, true);
      document.addEventListener("click", enlargeFigure);
    });
    onBeforeUnmount(() => {
      document.removeEventListener("keydown", restoreSearchFocus, true);
      document.removeEventListener("click", enlargeFigure);
      figureDialog?.remove();
    });

    return () =>
      h(DefaultTheme.Layout, null, {
        // Every chapter opens under a running head naming its part and position.
        "doc-before": () => h(ChapterOpener),
      });
  },
};

export default {
  extends: DefaultTheme,
  Layout: BookLayout,
  enhanceApp({ app }) {
    app.component("LandingPage", LandingPage);
  },
} satisfies Theme;
