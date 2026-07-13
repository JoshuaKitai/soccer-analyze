declare module "plotly.js-dist-min" {
  const Plotly: {
    newPlot: (
      el: HTMLElement | string,
      traces: unknown[],
      layout?: Record<string, unknown>,
      config?: Record<string, unknown>,
    ) => Promise<void>;
    restyle: (
      el: HTMLElement | string,
      update: Record<string, unknown>,
      traces?: number[],
    ) => Promise<void>;
    purge: (el: HTMLElement | string) => void;
    Plots: { resize: (el: HTMLElement) => void };
  };
  export default Plotly;
}
