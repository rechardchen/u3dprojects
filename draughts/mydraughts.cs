using Godot;
using System;


class Tile
{
    public Tile(int row, int col)
    {
        this.row= row;
        this.col= col;

        colorRect = new();
        colorRect.Size = new Vector2(mydraughts.TileSize,mydraughts.TileSize);
        colorRect.Position = new Vector2(col*mydraughts.TileSize,row*mydraughts.TileSize);
        if ((row + col) % 2 == 0)
        {
            colorRect.Color = mydraughts.WhiteColor;
        }
        else
        {
            colorRect.Color = mydraughts.BlackColor;
            label = new Label();
            label.Text = tid.ToString();
            label.AddThemeColorOverride("font_color", new Color(0, 0, 0));
            colorRect.AddChild(label);
        }
    }

    public int tid => (row * mydraughts.BoardSize + col) / 2 + 1;

    public ColorRect colorRect;
    private Label label;

    private int row = 0;
    private int col = 0;
}

public partial class mydraughts : Node
{
    public static int BoardSize = 10;

    public static int TileSize = 96;

    public static Color WhiteColor = new Color(0.99f, 0.96f, 0.9f);
    public static Color BlackColor = new Color(0.95f, 0.64f, 0.37f);

    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
    {
        GetWindow().Size = new Vector2I(BoardSize*TileSize, BoardSize*TileSize);

        for (int row = 0; row < BoardSize; row++)
        {
            for (int col = 0; col < BoardSize; col++)
            {
                var tile = new Tile(row, col);

                AddChild(tile.colorRect);
            }
        }

        GD.Print("Draughts initilalized...");
    }

    // Called every frame. 'delta' is the elapsed time since the previous frame.
    public override void _Process(double delta)
    {
    }
}
