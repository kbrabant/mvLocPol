function val = getopt(s, name, default)
    if isfield(s,name)
        val = s.(name);
    else
        val = default;
    end
end

