Sub MakeProgressBar()
    Const r As String = "00"    '色・RGB値のR
    Const g As String = "99"    '色・RGB値のG
    Const b As String = "00"    '色・RGB値のB
    Const pbH As Long = 10     '高さ
    Const pbBG As Single = 0.6  '背景の透過性

    Dim i As Long
    Dim s As Shape

    Dim wTop As Long 'プログレスバー位置
    Dim rc As Integer

    On Error Resume Next

    rc = MsgBox("プログレスバー位置はどこにしますか？" & vbCrLf & "上部（はい）　下部（いいえ）", vbYesNo + vbQuestion, "確認")
    If rc = vbYes Then
        wTop = 0
    Else
        wTop = ActivePresentation.PageSetup.SlideHeight - pbH
    End If

    With ActivePresentation
        '背景 ProgressBarBG の設定
        .SlideMaster.Shapes("ProgressBarBG").Delete
        Set s = .SlideMaster.Shapes.AddShape( _
        Type:=msoShapeRectangle, _
        Left:=0, _
        Height:=pbH, _
        Top:=wTop, _
        Width:=.PageSetup.SlideWidth)
        With s
            .Fill.ForeColor.RGB = _
            RGB(CInt("&H" & r), CInt("&H" & g), CInt("&H" & b))
            .Fill.Transparency = pbBG
            .Line.Visible = msoFalse
            .Name = "ProgressBarBG"
        End With

        'プログレスバー ProgressBar の設定
        For i = 1 To .Slides.Count
            .Slides(i).Shapes("ProgressBar").Delete
            Set s = .Slides(i).Shapes.AddShape( _
            Type:=msoShapeRectangle, _
            Left:=0, _
            Height:=pbH, _
            Top:=wTop, _
            Width:=i * .PageSetup.SlideWidth / .Slides.Count)
            With s
                .Fill.ForeColor.RGB = _
                RGB(CInt("&H" & r), CInt("&H" & g), CInt("&H" & b))
                .Line.Visible = msoFalse
                .Name = "ProgressBar"
            End With
        Next i
    End With

End Sub
