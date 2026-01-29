import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/app/components/ui/card";
import { Button } from "@/app/components/ui/button";
import { Download, Info } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

type TimeRange = "1Y" | "3Y" | "5Y" | "All";

export function MarketDataPage() {
  const [timeRange, setTimeRange] = useState<TimeRange>("1Y");

  // Mock data for Search Engine Market Share
  const searchEngineData = [
    { month: "Jan", Google: 92.3, Bing: 3.1, Yahoo: 1.5, Others: 3.1 },
    { month: "Feb", Google: 92.5, Bing: 3.0, Yahoo: 1.4, Others: 3.1 },
    { month: "Mar", Google: 92.1, Bing: 3.2, Yahoo: 1.6, Others: 3.1 },
    { month: "Apr", Google: 91.9, Bing: 3.4, Yahoo: 1.5, Others: 3.2 },
    { month: "May", Google: 92.0, Bing: 3.3, Yahoo: 1.5, Others: 3.2 },
    { month: "Jun", Google: 91.8, Bing: 3.5, Yahoo: 1.5, Others: 3.2 },
    { month: "Jul", Google: 91.7, Bing: 3.6, Yahoo: 1.4, Others: 3.3 },
    { month: "Aug", Google: 91.5, Bing: 3.8, Yahoo: 1.4, Others: 3.3 },
    { month: "Sep", Google: 91.4, Bing: 3.9, Yahoo: 1.4, Others: 3.3 },
    { month: "Oct", Google: 91.3, Bing: 4.0, Yahoo: 1.4, Others: 3.3 },
    { month: "Nov", Google: 91.2, Bing: 4.1, Yahoo: 1.4, Others: 3.3 },
    { month: "Dec", Google: 91.0, Bing: 4.3, Yahoo: 1.4, Others: 3.3 },
  ];

  // Mock data for Global Market Indices
  const marketIndicesData = [
    { month: "Jan", "S&P 500": 4500, FTSE: 7800, Nikkei: 33000, "DAX": 16500 },
    { month: "Feb", "S&P 500": 4550, FTSE: 7850, Nikkei: 33500, "DAX": 16700 },
    { month: "Mar", "S&P 500": 4600, FTSE: 7900, Nikkei: 34000, "DAX": 16900 },
    { month: "Apr", "S&P 500": 4650, FTSE: 7950, Nikkei: 34500, "DAX": 17100 },
    { month: "May", "S&P 500": 4700, FTSE: 8000, Nikkei: 35000, "DAX": 17300 },
    { month: "Jun", "S&P 500": 4750, FTSE: 8050, Nikkei: 35500, "DAX": 17500 },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-semibold text-gray-900">Market Data</h1>
        <p className="text-gray-600 mt-1">Global micro & macro market indicators</p>
      </div>

      {/* Key Indicators Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="border">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-gray-500 uppercase tracking-wide">S&P 500</p>
                <p className="text-2xl font-semibold mt-1">4,783.45</p>
                <p className="text-sm text-green-600 mt-1">+38.67 (+0.8%)</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-gray-500 uppercase tracking-wide">NASDAQ</p>
                <p className="text-2xl font-semibold mt-1">15,095.14</p>
                <p className="text-sm text-green-600 mt-1">+179.35 (+1.2%)</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-gray-500 uppercase tracking-wide">VIX</p>
                <p className="text-2xl font-semibold mt-1">13.42</p>
                <p className="text-sm text-red-600 mt-1">-0.29 (-2.1%)</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-gray-500 uppercase tracking-wide">10Y Treasury</p>
                <p className="text-2xl font-semibold mt-1">4.28%</p>
                <p className="text-sm text-green-600 mt-1">+0.05 bps</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Search Engine Market Share Chart */}
      <Card className="border shadow-sm">
        <CardHeader>
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center space-x-2">
                <CardTitle>Search Engine Market Share</CardTitle>
                <Info className="w-4 h-4 text-gray-400" />
              </div>
              <CardDescription className="mt-1">
                Market share percentage by search engine (Last 12 months)
              </CardDescription>
            </div>
            <div className="flex items-center space-x-2">
              <div className="flex items-center bg-gray-100 rounded-md p-1">
                {(["1Y", "3Y", "5Y", "All"] as TimeRange[]).map((range) => (
                  <Button
                    key={range}
                    variant="ghost"
                    size="sm"
                    className={`px-3 h-7 text-xs ${
                      timeRange === range
                        ? "bg-white shadow-sm"
                        : "hover:bg-transparent"
                    }`}
                    onClick={() => setTimeRange(range)}
                  >
                    {range}
                  </Button>
                ))}
              </div>
              <Button variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={searchEngineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="month" stroke="#888" fontSize={12} />
                <YAxis stroke="#888" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "white",
                    border: "1px solid #e5e7eb",
                    borderRadius: "6px",
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: "20px" }} />
                <Bar dataKey="Google" stackId="a" fill="#4285F4" />
                <Bar dataKey="Bing" stackId="a" fill="#00A4EF" />
                <Bar dataKey="Yahoo" stackId="a" fill="#6001D2" />
                <Bar dataKey="Others" stackId="a" fill="#9CA3AF" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 text-xs text-gray-500">
            <p>Data Source: StatCounter Global Stats • Last updated: Jan 29, 2026</p>
          </div>
        </CardContent>
      </Card>

      {/* Global Market Indices Chart */}
      <Card className="border shadow-sm">
        <CardHeader>
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center space-x-2">
                <CardTitle>Global Market Indices</CardTitle>
                <Info className="w-4 h-4 text-gray-400" />
              </div>
              <CardDescription className="mt-1">
                Performance of major global indices (YTD 2026)
              </CardDescription>
            </div>
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={marketIndicesData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="month" stroke="#888" fontSize={12} />
                <YAxis stroke="#888" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "white",
                    border: "1px solid #e5e7eb",
                    borderRadius: "6px",
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: "20px" }} />
                <Bar dataKey="S&P 500" stackId="a" fill="#10B981" />
                <Bar dataKey="FTSE" stackId="a" fill="#3B82F6" />
                <Bar dataKey="Nikkei" stackId="a" fill="#F59E0B" />
                <Bar dataKey="DAX" stackId="a" fill="#8B5CF6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 text-xs text-gray-500">
            <p>Data Source: Bloomberg Terminal • Last updated: Jan 29, 2026</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
