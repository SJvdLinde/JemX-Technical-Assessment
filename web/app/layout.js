import "./globals.css";

export const metadata = {
  title: "Ops Room — overtime by Sunday",
  description: "Who will breach the 10-hour overtime cap, and what to do today.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
