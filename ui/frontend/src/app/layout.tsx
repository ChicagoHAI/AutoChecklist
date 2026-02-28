import type { Metadata } from "next";
import { Inconsolata, Space_Mono } from "next/font/google";
import "./globals.css";
import { TopNav } from "@/components/TopNav";
import { QueryProvider } from "@/lib/query-provider";
import { OnboardingDialog } from "@/components/OnboardingDialog";
import { TooltipProvider } from "@/components/ui/Tooltip";

const inconsolata = Inconsolata({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
  variable: "--font-inconsolata",
});

const spaceMono = Space_Mono({
  subsets: ["latin"],
  weight: ["400", "700"],
  variable: "--font-space-mono",
});

export const metadata: Metadata = {
  title: "AutoChecklist",
  description: "Automatically evaluate outputs using LLM-based checklist methods.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inconsolata.variable} ${spaceMono.variable}`}>
      <body className="antialiased min-h-screen">
        <QueryProvider>
          <TooltipProvider>
            <TopNav />
            <main className="flex-1 overflow-y-auto">
              <div className="max-w-7xl mx-auto px-6 sm:px-10 lg:px-16 py-10">
                {children}
              </div>
            </main>
            <OnboardingDialog />
          </TooltipProvider>
        </QueryProvider>
      </body>
    </html>
  );
}
