import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/app/components/ui/card";
import { Button } from "@/app/components/ui/button";
import { Settings, TrendingUp, Activity, Calendar, DollarSign } from "lucide-react";
import { Link } from "react-router";
import { Badge } from "@/app/components/ui/badge";

interface Widget {
  id: string;
  title: string;
  type: "market-data" | "active-etf" | "earnings" | "stats";
  visible: boolean;
}

export function HomePage() {
  const [widgets, setWidgets] = useState<Widget[]>([
    { id: "market-overview", title: "Market Overview", type: "market-data", visible: true },
    { id: "etf-changes", title: "Active ETF Changes", type: "active-etf", visible: true },
    { id: "upcoming-earnings", title: "Upcoming Earnings", type: "earnings", visible: true },
    { id: "key-metrics", title: "Key Metrics", type: "stats", visible: true },
  ]);

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-semibold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-1">Overview of market insights and portfolio tracking</p>
        </div>
        <Button variant="outline" size="sm">
          <Settings className="w-4 h-4 mr-2" />
          Customize
        </Button>
      </div>

      {/* Widgets Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Market Overview Widget */}
        {widgets.find(w => w.id === "market-overview")?.visible && (
          <Card className="border shadow-sm">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <TrendingUp className="w-5 h-5 text-blue-600" />
                  <CardTitle className="text-lg">Market Overview</CardTitle>
                </div>
                <Link to="/market-data">
                  <Button variant="ghost" size="sm" className="text-blue-600">
                    View All →
                  </Button>
                </Link>
              </div>
              <CardDescription>Global macro & micro indicators</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center justify-between py-2 border-b">
                  <span className="text-sm text-gray-600">S&P 500</span>
                  <div className="text-right">
                    <span className="text-sm font-medium">4,783.45</span>
                    <span className="text-green-600 text-sm ml-2">+0.8%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between py-2 border-b">
                  <span className="text-sm text-gray-600">NASDAQ</span>
                  <div className="text-right">
                    <span className="text-sm font-medium">15,095.14</span>
                    <span className="text-green-600 text-sm ml-2">+1.2%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between py-2 border-b">
                  <span className="text-sm text-gray-600">VIX</span>
                  <div className="text-right">
                    <span className="text-sm font-medium">13.42</span>
                    <span className="text-red-600 text-sm ml-2">-2.1%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between py-2">
                  <span className="text-sm text-gray-600">10Y Treasury</span>
                  <div className="text-right">
                    <span className="text-sm font-medium">4.28%</span>
                    <span className="text-green-600 text-sm ml-2">+0.05</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Active ETF Changes Widget */}
        {widgets.find(w => w.id === "etf-changes")?.visible && (
          <Card className="border shadow-sm">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Activity className="w-5 h-5 text-purple-600" />
                  <CardTitle className="text-lg">Active ETF Changes</CardTitle>
                </div>
                <Link to="/active-etf">
                  <Button variant="ghost" size="sm" className="text-purple-600">
                    View All →
                  </Button>
                </Link>
              </div>
              <CardDescription>Recent portfolio rebalancing</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-start justify-between py-2 border-b">
                  <div>
                    <p className="text-sm font-medium">ARKK</p>
                    <p className="text-xs text-gray-500">ARK Innovation ETF</p>
                  </div>
                  <div className="text-right">
                    <Badge variant="outline" className="text-xs text-green-600 border-green-600">
                      +2 Additions
                    </Badge>
                  </div>
                </div>
                <div className="flex items-start justify-between py-2 border-b">
                  <div>
                    <p className="text-sm font-medium">QQQM</p>
                    <p className="text-xs text-gray-500">Invesco NASDAQ 100 ETF</p>
                  </div>
                  <div className="text-right">
                    <Badge variant="outline" className="text-xs text-blue-600 border-blue-600">
                      Weight ↑
                    </Badge>
                  </div>
                </div>
                <div className="flex items-start justify-between py-2">
                  <div>
                    <p className="text-sm font-medium">SCHD</p>
                    <p className="text-xs text-gray-500">Schwab US Dividend Equity ETF</p>
                  </div>
                  <div className="text-right">
                    <Badge variant="outline" className="text-xs text-red-600 border-red-600">
                      -1 Deletion
                    </Badge>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Upcoming Earnings Widget */}
        {widgets.find(w => w.id === "upcoming-earnings")?.visible && (
          <Card className="border shadow-sm">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Calendar className="w-5 h-5 text-orange-600" />
                  <CardTitle className="text-lg">Upcoming Earnings</CardTitle>
                </div>
                <Link to="/earnings-trading">
                  <Button variant="ghost" size="sm" className="text-orange-600">
                    View All →
                  </Button>
                </Link>
              </div>
              <CardDescription>Next 7 days</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center justify-between py-2 border-b">
                  <div>
                    <p className="text-sm font-medium">AAPL</p>
                    <p className="text-xs text-gray-500">Feb 1 • AMC</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm">$2.18</p>
                    <p className="text-xs text-gray-500">Consensus EPS</p>
                  </div>
                </div>
                <div className="flex items-center justify-between py-2 border-b">
                  <div>
                    <p className="text-sm font-medium">MSFT</p>
                    <p className="text-xs text-gray-500">Feb 2 • AMC</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm">$2.78</p>
                    <p className="text-xs text-gray-500">Consensus EPS</p>
                  </div>
                </div>
                <div className="flex items-center justify-between py-2 border-b">
                  <div>
                    <p className="text-sm font-medium">GOOGL</p>
                    <p className="text-xs text-gray-500">Feb 3 • BMO</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm">$1.63</p>
                    <p className="text-xs text-gray-500">Consensus EPS</p>
                  </div>
                </div>
                <div className="flex items-center justify-between py-2">
                  <div>
                    <p className="text-sm font-medium">AMZN</p>
                    <p className="text-xs text-gray-500">Feb 4 • AMC</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm">$1.12</p>
                    <p className="text-xs text-gray-500">Consensus EPS</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Key Metrics Widget */}
        {widgets.find(w => w.id === "key-metrics")?.visible && (
          <Card className="border shadow-sm">
            <CardHeader>
              <div className="flex items-center space-x-2">
                <DollarSign className="w-5 h-5 text-green-600" />
                <CardTitle className="text-lg">Key Metrics</CardTitle>
              </div>
              <CardDescription>Portfolio performance indicators</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <p className="text-xs text-gray-500">Active ETFs Monitored</p>
                  <p className="text-2xl font-semibold">3</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-gray-500">Holdings Tracked</p>
                  <p className="text-2xl font-semibold">147</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-gray-500">Daily Changes</p>
                  <p className="text-2xl font-semibold">12</p>
                  <p className="text-xs text-green-600">+5 vs. yesterday</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-gray-500">Earnings This Week</p>
                  <p className="text-2xl font-semibold">23</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
