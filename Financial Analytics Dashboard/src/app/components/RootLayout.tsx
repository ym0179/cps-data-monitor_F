import { Outlet, Link, useLocation } from "react-router";
import { Button } from "@/app/components/ui/button";

export function RootLayout() {
  const location = useLocation();
  
  const navItems = [
    { path: "/", label: "Home" },
    { path: "/market-data", label: "Market Data" },
    { path: "/active-etf", label: "Active ETF" },
    { path: "/earnings-trading", label: "Earnings Trading" },
  ];

  return (
    <div className="min-h-screen bg-white">
      {/* Top Navigation */}
      <header className="border-b bg-white sticky top-0 z-50">
        <div className="container mx-auto px-6">
          <div className="flex items-center justify-between h-16">
            {/* Logo/Brand */}
            <div className="flex items-center space-x-8">
              <h1 className="text-xl font-semibold text-gray-900">
                Analytics Platform
              </h1>
              
              {/* Navigation Links */}
              <nav className="hidden md:flex items-center space-x-1">
                {navItems.map((item) => (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                      location.pathname === item.path
                        ? "bg-gray-100 text-gray-900"
                        : "text-gray-600 hover:text-gray-900 hover:bg-gray-50"
                    }`}
                  >
                    {item.label}
                  </Link>
                ))}
              </nav>
            </div>

            {/* Login/Sign up */}
            <div className="flex items-center space-x-2">
              <Link to="/login">
                <Button variant="ghost" size="sm">
                  Login
                </Button>
              </Link>
              <Link to="/login">
                <Button size="sm">
                  Sign up
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <Outlet />
      </main>
    </div>
  );
}
