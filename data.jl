using OhMyJulia
using AbstractTrees
using Cascadia
using Gumbo
using Requests: get, text
using Fire

function get_pages()
    pages = []
    @sync for i in 1:30
        @async let data = get("http://jhsjk.people.cn/result/$i") |> text |> parsehtml
            x = matchall(sel"li > a[target='_blank']", data.root)
            append!(pages, map(x->x.attributes["href"], x))
        end
    end
    pages
end

function extract_presentation(f, page)
    data = get("http://jhsjk.people.cn/$page") |> text |> parsehtml
    content = matchall(sel".d2txt_con", data.root)[]
    for el in PreOrderDFS(content) @when el isa HTMLText
        print(f, el.text)
    end
end

@main function scrape()
    pages = get_pages()
    i, con = Ref(30), Condition()
    for p in pages
        i[] <= 0 && wait(con)
        i[] -= 1
        path = "D:/xi-presentation/raw/$(split(p, '/')[end]).txt"
        @schedule open(path, "w") do f
            extract_presentation(f, p)
            notify(con)
        end
    end
end

@main function clean()
    files = map(x->"D:/xi-presentation/raw/$x", readdir("D:/xi-presentation/raw"))
    text = mapreduce(readstring, *, files)

    dict = Dict(
        '0' => '０',
        '1' => '１',
        '2' => '２',
        '3' => '３',
        '4' => '４',
        '5' => '５',
        '6' => '６',
        '7' => '７',
        '8' => '８',
        '9' => '９',
        ',' => '，',
        '(' => '（',
        '﹝' => '（',
        ')' => '）',
        '﹞' => '）',
        '!' => '！',
        '?' => '？',
        ':' => '：',
        '　' => ' '
    )

    text = map(x->get(dict, x, x), text)
    text = replace(text, r"\t\s*", "")
    text = replace(text, r"【.{0,6}】", "")
    text = replace(text, r"[★▲◆●▌■◎○]", '-')
    
    vocabulary = groupby(x->x, (x,y)->x+1, 0, text)

    println("""\n
        total length: $(length(text))
        vocabulary size: $(length(vocabulary))
        # of words freq <= 2: $(count(x->x[2]<=2, vocabulary))
    """)
    
    open("D:/xi-presentation/clean/data.txt", "w") do f
        println(f, text)
    end
end