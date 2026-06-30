import type { ReactNode } from "react";
import { render } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

/** Render a view inside a router. If `path`/`route` are given, the element is
 *  mounted on that route so useParams() resolves. */
export function renderRoute(
  element: ReactNode,
  { path = "/", route = "/" }: { path?: string; route?: string } = {},
) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <Routes>
        <Route path={path} element={element} />
      </Routes>
    </MemoryRouter>,
  );
}
