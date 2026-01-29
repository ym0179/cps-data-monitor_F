import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";
import { Button } from "@/app/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/app/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/app/components/ui/tabs";
import { ArrowUp, ArrowDown, Plus, Minus, Sparkles } from "lucide-react";

type ChangeType = "addition" | "deletion" | "weight-increase" | "weight-decrease";

interface PortfolioChange {
  ticker: string;
  name: string;
  changeType: ChangeType;
  beforeWeight: number | null;
  afterWeight: number | null;
  sector: string;
}

interface Holding {
  rank: number;
  ticker: string;
  name: string;
  weight: number;
  change: number;
  sector: string;
}

export function ActiveETFPage() {
  const [selectedETF, setSelectedETF] = useState<"ARKK" | "QQQM" | "SCHD">("ARKK");

  const etfs = [
    { id: "ARKK", name: "ARK Innovation ETF", changesCount: 5 },
    { id: "QQQM", name: "Invesco NASDAQ 100 ETF", changesCount: 3 },
    { id: "SCHD", name: "Schwab US Dividend Equity ETF", changesCount: 2 },
  ];

  // Mock rebalancing data
  const rebalancingChanges: Record<string, PortfolioChange[]> = {
    ARKK: [
      {
        ticker: "NVDA",
        name: "NVIDIA Corporation",
        changeType: "addition",
        beforeWeight: null,
        afterWeight: 8.5,
        sector: "Technology",
      },
      {
        ticker: "META",
        name: "Meta Platforms Inc",
        changeType: "weight-increase",
        beforeWeight: 6.2,
        afterWeight: 8.1,
        sector: "Technology",
      },
      {
        ticker: "SHOP",
        name: "Shopify Inc",
        changeType: "deletion",
        beforeWeight: 3.4,
        afterWeight: null,
        sector: "Technology",
      },
      {
        ticker: "ROKU",
        name: "Roku Inc",
        changeType: "weight-decrease",
        beforeWeight: 4.5,
        afterWeight: 2.8,
        sector: "Technology",
      },
      {
        ticker: "SQ",
        name: "Block Inc",
        changeType: "weight-decrease",
        beforeWeight: 5.1,
        afterWeight: 3.6,
        sector: "Financial Services",
      },
    ],
    QQQM: [
      {
        ticker: "AAPL",
        name: "Apple Inc",
        changeType: "weight-increase",
        beforeWeight: 11.2,
        afterWeight: 12.1,
        sector: "Technology",
      },
      {
        ticker: "MSFT",
        name: "Microsoft Corporation",
        changeType: "weight-increase",
        beforeWeight: 10.5,
        afterWeight: 11.3,
        sector: "Technology",
      },
      {
        ticker: "TSLA",
        name: "Tesla Inc",
        changeType: "weight-decrease",
        beforeWeight: 4.2,
        afterWeight: 3.5,
        sector: "Automotive",
      },
    ],
    SCHD: [
      {
        ticker: "VZ",
        name: "Verizon Communications",
        changeType: "deletion",
        beforeWeight: 3.1,
        afterWeight: null,
        sector: "Telecommunications",
      },
      {
        ticker: "HD",
        name: "The Home Depot Inc",
        changeType: "weight-increase",
        beforeWeight: 4.5,
        afterWeight: 5.2,
        sector: "Consumer Cyclical",
      },
    ],
  };

  // Mock top holdings data
  const topHoldings: Record<string, Holding[]> = {
    ARKK: [
      { rank: 1, ticker: "TSLA", name: "Tesla Inc", weight: 10.2, change: 0.3, sector: "Automotive" },
      { rank: 2, ticker: "COIN", name: "Coinbase Global", weight: 9.5, change: -0.5, sector: "Financial Services" },
      { rank: 3, ticker: "NVDA", name: "NVIDIA Corporation", weight: 8.5, change: 8.5, sector: "Technology" },
      { rank: 4, ticker: "META", name: "Meta Platforms", weight: 8.1, change: 1.9, sector: "Technology" },
      { rank: 5, ticker: "RBLX", name: "Roblox Corporation", weight: 7.3, change: 0.0, sector: "Technology" },
      { rank: 6, ticker: "PATH", name: "UiPath Inc", weight: 5.8, change: -0.2, sector: "Technology" },
      { rank: 7, ticker: "TDOC", name: "Teladoc Health", weight: 4.9, change: 0.1, sector: "Healthcare" },
      { rank: 8, ticker: "DKNG", name: "DraftKings Inc", weight: 4.2, change: 0.0, sector: "Consumer Cyclical" },
      { rank: 9, ticker: "SQ", name: "Block Inc", weight: 3.6, change: -1.5, sector: "Financial Services" },
      { rank: 10, ticker: "ROKU", name: "Roku Inc", weight: 2.8, change: -1.7, sector: "Technology" },
    ],
    QQQM: [
      { rank: 1, ticker: "AAPL", name: "Apple Inc", weight: 12.1, change: 0.9, sector: "Technology" },
      { rank: 2, ticker: "MSFT", name: "Microsoft Corporation", weight: 11.3, change: 0.8, sector: "Technology" },
      { rank: 3, ticker: "NVDA", name: "NVIDIA Corporation", weight: 8.4, change: 0.2, sector: "Technology" },
      { rank: 4, ticker: "AMZN", name: "Amazon.com Inc", weight: 7.2, change: 0.0, sector: "Consumer Cyclical" },
      { rank: 5, ticker: "GOOGL", name: "Alphabet Inc Class A", weight: 5.9, change: -0.1, sector: "Technology" },
      { rank: 6, ticker: "META", name: "Meta Platforms", weight: 5.1, change: 0.0, sector: "Technology" },
      { rank: 7, ticker: "TSLA", name: "Tesla Inc", weight: 3.5, change: -0.7, sector: "Automotive" },
      { rank: 8, ticker: "AVGO", name: "Broadcom Inc", weight: 3.2, change: 0.1, sector: "Technology" },
      { rank: 9, ticker: "COST", name: "Costco Wholesale", weight: 2.8, change: 0.0, sector: "Consumer Defensive" },
      { rank: 10, ticker: "NFLX", name: "Netflix Inc", weight: 2.5, change: -0.1, sector: "Communication Services" },
    ],
    SCHD: [
      { rank: 1, ticker: "HD", name: "The Home Depot", weight: 5.2, change: 0.7, sector: "Consumer Cyclical" },
      { rank: 2, ticker: "PEP", name: "PepsiCo Inc", weight: 4.8, change: 0.0, sector: "Consumer Defensive" },
      { rank: 3, ticker: "TXN", name: "Texas Instruments", weight: 4.5, change: 0.0, sector: "Technology" },
      { rank: 4, ticker: "JNJ", name: "Johnson & Johnson", weight: 4.3, change: 0.1, sector: "Healthcare" },
      { rank: 5, ticker: "CVX", name: "Chevron Corporation", weight: 4.1, change: -0.1, sector: "Energy" },
      { rank: 6, ticker: "PFE", name: "Pfizer Inc", weight: 3.9, change: 0.0, sector: "Healthcare" },
      { rank: 7, ticker: "BMY", name: "Bristol Myers Squibb", weight: 3.7, change: 0.1, sector: "Healthcare" },
      { rank: 8, ticker: "AMGN", name: "Amgen Inc", weight: 3.5, change: 0.0, sector: "Healthcare" },
      { rank: 9, ticker: "MRK", name: "Merck & Co", weight: 3.4, change: 0.0, sector: "Healthcare" },
      { rank: 10, ticker: "KO", name: "The Coca-Cola Company", weight: 3.2, change: 0.0, sector: "Consumer Defensive" },
    ],
  };

  const aiCommentary: Record<string, string> = {
    ARKK: "ARKK's latest rebalancing shows a strategic pivot toward established AI infrastructure plays, with NVIDIA's addition marking a significant shift from pure disruptive growth to proven AI leaders. The fund increased exposure to Meta while trimming positions in struggling e-commerce and fintech names. The weight reduction in Roku and Block suggests profit-taking after recent rallies, while the complete exit from Shopify may signal concerns about e-commerce growth headwinds.",
    QQQM: "QQQM's quarterly rebalance reflects the index's market-cap weighting methodology, with Apple and Microsoft seeing modest weight increases following strong Q4 performance. The Tesla reduction is purely mechanical, driven by market cap changes relative to other holdings. Overall portfolio adjustments are minimal, consistent with the fund's passive indexing approach to the top 100 NASDAQ stocks.",
    SCHD: "SCHD's rebalancing maintains its focus on quality dividend payers, exiting Verizon following recent dividend sustainability concerns. The increased allocation to Home Depot reflects the stock's strong fundamental performance and consistent dividend growth. Portfolio changes align with the fund's methodology of screening for dividend sustainability, quality metrics, and low volatility characteristics.",
  };

  const getChangeIcon = (type: ChangeType) => {
    switch (type) {
      case "addition":
        return <Plus className="w-4 h-4 text-green-600" />;
      case "deletion":
        return <Minus className="w-4 h-4 text-red-600" />;
      case "weight-increase":
        return <ArrowUp className="w-4 h-4 text-blue-600" />;
      case "weight-decrease":
        return <ArrowDown className="w-4 h-4 text-orange-600" />;
    }
  };

  const getChangeBadge = (type: ChangeType) => {
    switch (type) {
      case "addition":
        return <Badge className="bg-green-50 text-green-700 border-green-200">New Position</Badge>;
      case "deletion":
        return <Badge className="bg-red-50 text-red-700 border-red-200">Removed</Badge>;
      case "weight-increase":
        return <Badge className="bg-blue-50 text-blue-700 border-blue-200">Weight ↑</Badge>;
      case "weight-decrease":
        return <Badge className="bg-orange-50 text-orange-700 border-orange-200">Weight ↓</Badge>;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-semibold text-gray-900">Active ETF Monitoring</h1>
        <p className="text-gray-600 mt-1">Track portfolio changes and rebalancing activity</p>
      </div>

      {/* ETF Selection Tabs */}
      <Tabs value={selectedETF} onValueChange={(v) => setSelectedETF(v as any)} className="w-full">
        <TabsList className="grid w-full max-w-2xl grid-cols-3">
          {etfs.map((etf) => (
            <TabsTrigger key={etf.id} value={etf.id} className="relative">
              {etf.name}
              {etf.changesCount > 0 && (
                <span className="ml-2 bg-blue-600 text-white text-xs px-2 py-0.5 rounded-full">
                  {etf.changesCount}
                </span>
              )}
            </TabsTrigger>
          ))}
        </TabsList>

        {etfs.map((etf) => (
          <TabsContent key={etf.id} value={etf.id} className="space-y-6 mt-6">
            {/* Daily Rebalancing Summary */}
            <Card className="border shadow-sm">
              <CardHeader>
                <CardTitle>Daily Rebalancing Summary</CardTitle>
                <CardDescription>Changes detected on Jan 29, 2026</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {rebalancingChanges[etf.id].map((change) => (
                    <div
                      key={change.ticker}
                      className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex items-center space-x-3">
                        {getChangeIcon(change.changeType)}
                        <div>
                          <div className="flex items-center space-x-2">
                            <span className="font-medium">{change.ticker}</span>
                            <span className="text-sm text-gray-500">{change.name}</span>
                          </div>
                          <div className="flex items-center space-x-2 mt-1">
                            <span className="text-xs text-gray-500">{change.sector}</span>
                            {getChangeBadge(change.changeType)}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        {change.beforeWeight !== null && change.afterWeight !== null && (
                          <div className="text-sm">
                            <span className="text-gray-400">{change.beforeWeight.toFixed(1)}%</span>
                            <span className="mx-2">→</span>
                            <span className="font-medium">{change.afterWeight.toFixed(1)}%</span>
                          </div>
                        )}
                        {change.beforeWeight === null && change.afterWeight !== null && (
                          <div className="text-sm">
                            <span className="font-medium text-green-600">
                              {change.afterWeight.toFixed(1)}%
                            </span>
                          </div>
                        )}
                        {change.beforeWeight !== null && change.afterWeight === null && (
                          <div className="text-sm">
                            <span className="text-gray-400 line-through">
                              {change.beforeWeight.toFixed(1)}%
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* AI Commentary */}
            <Card className="border shadow-sm bg-gradient-to-br from-blue-50 to-indigo-50">
              <CardHeader>
                <div className="flex items-center space-x-2">
                  <Sparkles className="w-5 h-5 text-blue-600" />
                  <CardTitle>AI Analysis</CardTitle>
                </div>
                <CardDescription>AI-generated insights on portfolio changes</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-700 leading-relaxed">{aiCommentary[etf.id]}</p>
              </CardContent>
            </Card>

            {/* Top 10 Holdings */}
            <Card className="border shadow-sm">
              <CardHeader>
                <CardTitle>Top 10 Holdings</CardTitle>
                <CardDescription>Current portfolio composition</CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-12">Rank</TableHead>
                      <TableHead>Ticker</TableHead>
                      <TableHead>Company Name</TableHead>
                      <TableHead>Sector</TableHead>
                      <TableHead className="text-right">Weight</TableHead>
                      <TableHead className="text-right">Change</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {topHoldings[etf.id].map((holding) => (
                      <TableRow key={holding.ticker}>
                        <TableCell className="font-medium">{holding.rank}</TableCell>
                        <TableCell className="font-mono font-medium">{holding.ticker}</TableCell>
                        <TableCell>{holding.name}</TableCell>
                        <TableCell>
                          <span className="text-xs text-gray-600">{holding.sector}</span>
                        </TableCell>
                        <TableCell className="text-right font-medium">
                          {holding.weight.toFixed(1)}%
                        </TableCell>
                        <TableCell className="text-right">
                          {holding.change !== 0 && (
                            <span
                              className={`text-sm ${
                                holding.change > 0 ? "text-green-600" : "text-red-600"
                              }`}
                            >
                              {holding.change > 0 ? "+" : ""}
                              {holding.change.toFixed(1)}%
                            </span>
                          )}
                          {holding.change === 0 && (
                            <span className="text-sm text-gray-400">—</span>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
