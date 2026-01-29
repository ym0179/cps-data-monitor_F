import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/app/components/ui/card";
import { Input } from "@/app/components/ui/input";
import { Badge } from "@/app/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/app/components/ui/table";
import { Search, TrendingUp, TrendingDown, Minus } from "lucide-react";

interface EarningsEvent {
  ticker: string;
  companyName: string;
  date: string;
  time: "BMO" | "AMC";
  marketCap: string;
  consensusEPS: number;
  impliedMove: number;
  positioningScore: number; // -100 to 100
  sector: string;
}

type PositioningLevel = "Bullish" | "Neutral" | "Bearish";

export function EarningsCalendarPage() {
  const [searchTerm, setSearchTerm] = useState("");

  const earningsData: EarningsEvent[] = [
    {
      ticker: "AAPL",
      companyName: "Apple Inc",
      date: "Feb 1",
      time: "AMC",
      marketCap: "$2.8T",
      consensusEPS: 2.18,
      impliedMove: 4.2,
      positioningScore: 65,
      sector: "Technology",
    },
    {
      ticker: "MSFT",
      companyName: "Microsoft Corporation",
      date: "Feb 2",
      time: "AMC",
      marketCap: "$2.9T",
      consensusEPS: 2.78,
      impliedMove: 3.8,
      positioningScore: 72,
      sector: "Technology",
    },
    {
      ticker: "GOOGL",
      companyName: "Alphabet Inc Class A",
      date: "Feb 3",
      time: "BMO",
      marketCap: "$1.7T",
      consensusEPS: 1.63,
      impliedMove: 5.1,
      positioningScore: 48,
      sector: "Technology",
    },
    {
      ticker: "AMZN",
      companyName: "Amazon.com Inc",
      date: "Feb 4",
      time: "AMC",
      marketCap: "$1.6T",
      consensusEPS: 1.12,
      impliedMove: 6.3,
      positioningScore: 55,
      sector: "Consumer Cyclical",
    },
    {
      ticker: "META",
      companyName: "Meta Platforms Inc",
      date: "Feb 4",
      time: "AMC",
      marketCap: "$1.2T",
      consensusEPS: 5.23,
      impliedMove: 7.8,
      positioningScore: 80,
      sector: "Technology",
    },
    {
      ticker: "NVDA",
      companyName: "NVIDIA Corporation",
      date: "Feb 5",
      time: "AMC",
      marketCap: "$1.8T",
      consensusEPS: 5.56,
      impliedMove: 9.2,
      positioningScore: 85,
      sector: "Technology",
    },
    {
      ticker: "TSLA",
      companyName: "Tesla Inc",
      date: "Feb 5",
      time: "AMC",
      marketCap: "$650B",
      consensusEPS: 0.73,
      impliedMove: 11.5,
      positioningScore: -25,
      sector: "Automotive",
    },
    {
      ticker: "JPM",
      companyName: "JPMorgan Chase & Co",
      date: "Feb 6",
      time: "BMO",
      marketCap: "$535B",
      consensusEPS: 3.97,
      impliedMove: 3.2,
      positioningScore: 15,
      sector: "Financial Services",
    },
    {
      ticker: "BAC",
      companyName: "Bank of America Corp",
      date: "Feb 6",
      time: "BMO",
      marketCap: "$310B",
      consensusEPS: 0.82,
      impliedMove: 3.5,
      positioningScore: 8,
      sector: "Financial Services",
    },
    {
      ticker: "JNJ",
      companyName: "Johnson & Johnson",
      date: "Feb 7",
      time: "BMO",
      marketCap: "$385B",
      consensusEPS: 2.35,
      impliedMove: 2.1,
      positioningScore: -5,
      sector: "Healthcare",
    },
    {
      ticker: "UNH",
      companyName: "UnitedHealth Group",
      date: "Feb 7",
      time: "BMO",
      marketCap: "$480B",
      consensusEPS: 6.45,
      impliedMove: 3.8,
      positioningScore: 28,
      sector: "Healthcare",
    },
    {
      ticker: "WMT",
      companyName: "Walmart Inc",
      date: "Feb 7",
      time: "BMO",
      marketCap: "$425B",
      consensusEPS: 1.68,
      impliedMove: 2.9,
      positioningScore: 42,
      sector: "Consumer Defensive",
    },
  ];

  const getPositioningLevel = (score: number): PositioningLevel => {
    if (score >= 50) return "Bullish";
    if (score <= -50) return "Bearish";
    return "Neutral";
  };

  const getPositioningColor = (level: PositioningLevel) => {
    switch (level) {
      case "Bullish":
        return "text-green-600 bg-green-50 border-green-200";
      case "Neutral":
        return "text-gray-600 bg-gray-50 border-gray-200";
      case "Bearish":
        return "text-red-600 bg-red-50 border-red-200";
    }
  };

  const getPositioningIcon = (level: PositioningLevel) => {
    switch (level) {
      case "Bullish":
        return <TrendingUp className="w-3 h-3" />;
      case "Neutral":
        return <Minus className="w-3 h-3" />;
      case "Bearish":
        return <TrendingDown className="w-3 h-3" />;
    }
  };

  const filteredEarnings = earningsData.filter(
    (event) =>
      event.ticker.toLowerCase().includes(searchTerm.toLowerCase()) ||
      event.companyName.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-semibold text-gray-900">Earnings Calendar</h1>
        <p className="text-gray-600 mt-1">Upcoming earnings reports with analytics</p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="border">
          <CardContent className="p-4">
            <p className="text-xs text-gray-500 uppercase tracking-wide">This Week</p>
            <p className="text-2xl font-semibold mt-1">12</p>
            <p className="text-sm text-gray-600 mt-1">Companies reporting</p>
          </CardContent>
        </Card>

        <Card className="border">
          <CardContent className="p-4">
            <p className="text-xs text-gray-500 uppercase tracking-wide">Mega Caps</p>
            <p className="text-2xl font-semibold mt-1">6</p>
            <p className="text-sm text-gray-600 mt-1">Over $1T market cap</p>
          </CardContent>
        </Card>

        <Card className="border">
          <CardContent className="p-4">
            <p className="text-xs text-gray-500 uppercase tracking-wide">Avg Implied Move</p>
            <p className="text-2xl font-semibold mt-1">5.2%</p>
            <p className="text-sm text-gray-600 mt-1">Options-implied volatility</p>
          </CardContent>
        </Card>

        <Card className="border">
          <CardContent className="p-4">
            <p className="text-xs text-gray-500 uppercase tracking-wide">Bullish Positioning</p>
            <p className="text-2xl font-semibold mt-1">58%</p>
            <p className="text-sm text-gray-600 mt-1">Of tracked names</p>
          </CardContent>
        </Card>
      </div>

      {/* Earnings Table */}
      <Card className="border shadow-sm">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Earnings Events</CardTitle>
              <CardDescription className="mt-1">
                Next 7 days • {filteredEarnings.length} companies
              </CardDescription>
            </div>
            <div className="relative w-64">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <Input
                type="text"
                placeholder="Search by ticker or company..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-9"
              />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Ticker</TableHead>
                <TableHead>Company</TableHead>
                <TableHead>Date</TableHead>
                <TableHead>Time</TableHead>
                <TableHead className="text-right">Market Cap</TableHead>
                <TableHead className="text-right">Consensus EPS</TableHead>
                <TableHead className="text-right">Implied Move</TableHead>
                <TableHead className="text-center">Positioning</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredEarnings.map((event) => {
                const level = getPositioningLevel(event.positioningScore);
                return (
                  <TableRow key={event.ticker} className="hover:bg-gray-50">
                    <TableCell className="font-mono font-medium">{event.ticker}</TableCell>
                    <TableCell>
                      <div>
                        <p className="font-medium">{event.companyName}</p>
                        <p className="text-xs text-gray-500">{event.sector}</p>
                      </div>
                    </TableCell>
                    <TableCell>{event.date}</TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className={
                          event.time === "BMO"
                            ? "bg-blue-50 text-blue-700 border-blue-200"
                            : "bg-purple-50 text-purple-700 border-purple-200"
                        }
                      >
                        {event.time}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right font-medium">
                      {event.marketCap}
                    </TableCell>
                    <TableCell className="text-right font-medium">
                      ${event.consensusEPS.toFixed(2)}
                    </TableCell>
                    <TableCell className="text-right">
                      <span className="font-medium">±{event.impliedMove.toFixed(1)}%</span>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center justify-center">
                        <Badge
                          variant="outline"
                          className={`${getPositioningColor(level)} flex items-center space-x-1`}
                        >
                          {getPositioningIcon(level)}
                          <span className="text-xs">{level}</span>
                        </Badge>
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Legend */}
      <Card className="border bg-gray-50">
        <CardHeader>
          <CardTitle className="text-sm">Legend</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <p className="font-medium mb-2">Report Timing</p>
              <p className="text-xs text-gray-600">
                <span className="font-medium">BMO:</span> Before Market Open
              </p>
              <p className="text-xs text-gray-600 mt-1">
                <span className="font-medium">AMC:</span> After Market Close
              </p>
            </div>
            <div>
              <p className="font-medium mb-2">Implied Move</p>
              <p className="text-xs text-gray-600">
                Options-implied stock price movement percentage based on at-the-money straddle pricing
              </p>
            </div>
            <div>
              <p className="font-medium mb-2">Positioning Score</p>
              <p className="text-xs text-gray-600">
                Proprietary metric combining options flow, analyst sentiment, and institutional positioning
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
