// verovio (npm) には型が付いていないので、使う所だけ書く
declare module "verovio/wasm" {
  const createVerovioModule: () => Promise<unknown>;
  export default createVerovioModule;
}

declare module "verovio/esm" {
  export class VerovioToolkit {
    constructor(module: unknown);
    setOptions(options: Record<string, unknown>): void;
    loadData(data: string): boolean;
    getPageCount(): number;
    renderToSVG(page: number, xmlDeclaration?: boolean): string;
    redoLayout(): void;
    getLog(): string;
  }
}
