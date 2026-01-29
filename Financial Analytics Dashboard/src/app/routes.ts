import { createBrowserRouter } from "react-router";
import { RootLayout } from "@/app/components/RootLayout";
import { HomePage } from "@/app/pages/HomePage";
import { MarketDataPage } from "@/app/pages/MarketDataPage";
import { ActiveETFPage } from "@/app/pages/ActiveETFPage";
import { EarningsCalendarPage } from "@/app/pages/EarningsCalendarPage";
import { LoginPage } from "@/app/pages/LoginPage";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: RootLayout,
    children: [
      { index: true, Component: HomePage },
      { path: "market-data", Component: MarketDataPage },
      { path: "active-etf", Component: ActiveETFPage },
      { path: "earnings-trading", Component: EarningsCalendarPage },
      { path: "login", Component: LoginPage },
    ],
  },
]);
