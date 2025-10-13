# {{ project-name }}

A React-based management dashboard for monitoring and testing Opper AI agents in real-time.

React Router v7 application with Bun, TypeScript, Tailwind CSS, and shadcn/ui + Radix UI components.

## Usage

Management interface for the agent server.

### Environment Variables

- `VITE_API_BASE_URL` - Base URL of the agent API (default: http://localhost:3030)

### Development

The UI runs on port 3000 by default and expects the agent API to be running on port 3030.

## For Coding Agents

## Import Path Rules

**CRITICAL**: Always use these import aliases:

- `~/` = `app/` directory - components, routes, app lib files (agentClient, callStore)
- `@/` = project root - shared utilities

```tsx
// CORRECT
import { useCallStore } from "~/lib/callStore";     // app/lib/callStore.ts
import { EventCard } from "~/components/EventCard"; // app/components/EventCard.tsx
import { cn } from "@/lib/utils";                   // lib/utils.ts (ROOT, not app!)

// WRONG
import { cn } from "~/lib/utils";                   // utils is in ROOT lib/, not app/lib/
```

**Remember**: `cn` utility is always `@/lib/utils`, everything else is likely `~/`

## Dependencies
To add packages, use the MCP tool:
```bash
__polytope__run(module: {{ project-name }}-add-dependencies, args: {packages: "axios react-query"})
```

## Theming & Styling Guidelines

This template uses **shadcn/ui's theme system** which automatically adapts to light/dark mode based on system preferences.

### Use Theme-Aware Classes
Always use shadcn/ui's semantic color classes that automatically adapt to the current theme:

```tsx
// CORRECT - These work perfectly in both light and dark modes
<div className="bg-background text-foreground">
<div className="bg-card text-card-foreground">
<Button className="bg-primary text-primary-foreground">
<div className="border border-border">
```

### DON'T: Use Fixed Tailwind Colors
Never use standard Tailwind color classes - they break theme consistency:
```tsx
// WRONG - These will cause visibility issues
<div className="bg-white text-black">        // Breaks in dark mode
<div className="bg-gray-50 text-gray-900">   // No theme adaptation
```

### Working with Colors
When you need variations or special effects, use Tailwind's opacity modifiers:
```tsx
// Opacity modifiers maintain theme consistency
<div className="bg-primary/10">            // 10% opacity of primary color
<div className="text-muted-foreground/50"> // 50% opacity text
<div className="bg-gradient-to-r from-primary/20 to-secondary/20">
```

### ALWAYS Use shadcn/ui Components First
Never create custom HTML elements when shadcn components exist:

```tsx
// CORRECT - Use shadcn components
import { Button } from "~/components/ui/button";
<Button variant="destructive" onClick={handleClick}>Click me</Button>
<Button variant="outline">Cancel</Button>

// WRONG - Manual button styling (even with semantic colors)
<button className="px-4 py-2 bg-primary text-primary-foreground rounded hover:bg-primary/90">
```

**Available Components**: `Button`, `Card`, `Input`, `Textarea`, `Badge`, `Avatar`, `Dialog`, `Popover`, `Sheet`, `Switch`, `Separator`, `ScrollArea`, `DropdownMenu`, `Sonner`

### Key Rules
1. **Use shadcn components first** - Don't reinvent buttons, cards, inputs, etc.
2. **If you find yourself typing a color name (red, blue, gray, slate, etc.), STOP!** Use semantic classes instead.
3. **Exception**: Pure black/white for logos or truly neutral elements that must stay constant.
