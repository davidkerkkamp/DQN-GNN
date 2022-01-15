import sqlite3
from typing import List


class Repository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.enable_load_extension(True)
        self.conn.execute('SELECT load_extension("mod_spatialite")')

    def get_cursor(self) -> sqlite3.Cursor:
        if not isinstance(self.conn, sqlite3.Connection):
            self.connect()
        return self.conn.cursor()

    def close(self):
        if isinstance(self.conn, sqlite3.Connection):
            self.conn.close()

    def get_many(self, query: str):
        cur = self.get_cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        return rows

    def get_sewer_where_clause(self, min_constr_year: int = None, area: List = None):
        wheres = []
        if min_constr_year is not None:
            wheres.append(f'construction_year >= {min_constr_year}')

        if area is not None:
            if isinstance(area, list) or isinstance(area, tuple) and len(area) == 4:
                wheres.append(f'st_minx(geometry) > {area[0]}')
                wheres.append(f'st_maxx(geometry) < {area[1]}')
                wheres.append(f'st_miny(geometry) > {area[2]}')
                wheres.append(f'st_maxy(geometry) < {area[3]}')
            else:
                raise Exception("'area' must be a list or tuple with minx, maxx, miny and maxy")
        return f"WHERE {' AND '.join([w for w in wheres])}" if len(wheres) > 0 else ""

    def get_sewer_pipes(self, fields: List = None, min_constr_year: int = None, area: List = None):
        try:
            select = '*' if fields is None else ', '.join([f for f in fields])
        except TypeError as e:
            raise Exception("'fields' must be a list of field names to select")

        where = self.get_sewer_where_clause(min_constr_year=min_constr_year, area=area)
        query = f"SELECT {select}\nFROM 'sewer_pipes'\n{where};"
        print(f'Sewer pipe query:\n{query}')
        return self.get_many(query)

    def get_sewer_coordinates(self, min_constr_year: int = None, area: List = None):
        where = self.get_sewer_where_clause(min_constr_year=min_constr_year, area=area)
        query = f"""WITH sewers AS 
        (SELECT id AS id, st_pointn(geometry, 1) as p1, st_pointn(geometry, 2) as p2
        FROM sewer_pipes
        {where})
        SELECT sewers.id, st_x(sewers.p1) as x1, st_y(sewers.p1) as y1, st_x(sewers.p2) as x2, st_y(sewers.p2) as y2
        FROM sewers;"""
        print(f'Sewer pipe coordinates query:\n{query}')
        return self.get_many(query)

    def get_sewers_with_neighbours(self, range: int = 20, min_constr_year: int = None, area: List = None):
        where = self.get_sewer_where_clause(min_constr_year=min_constr_year, area=area)

        query = f"""WITH sewers AS (
                SELECT id AS id, st_pointn(geometry, 1) as p1, st_pointn(geometry, 2) as p2, geometry as geometry
                FROM sewer_pipes
                {where}
                )
            SELECT s1.id as s1_id, st_x(s1.p1) as s1_x1, st_y(s1.p1) as s1_y1, st_x(s1.p2) as s1_x2, st_y(s1.p2) as s1_y2,
                   s2.id as s2_id, st_x(s2.p1) as s2_x1, st_y(s2.p1) as s2_y1, st_x(s2.p2) as s2_x2, st_y(s2.p2) as s2_y2
            FROM sewers s1 INNER JOIN sewers s2 ON s1.id != s2.id
            WHERE PtDistWithin(st_pointn(s1.geometry, 1), st_pointn(s2.geometry, 1), {range}) OR
                  PtDistWithin(st_pointn(s1.geometry, 2), st_pointn(s2.geometry, 2), {range}) OR
                  PtDistWithin(st_pointn(s1.geometry, 1), st_pointn(s2.geometry, 2), {range}) OR
                  PtDistWithin(st_pointn(s1.geometry, 2), st_pointn(s2.geometry, 1), {range});"""
        print(f'Sewer pipe neighbours query:\n{query}')
        return self.get_many(query)

    def get_sewers_with_damage_classes(self, min_constr_year: int = None, area: List = None):
        where = self.get_sewer_where_clause(min_constr_year=min_constr_year, area=area)

        query = f"""
        SELECT sp.id as sewer_pipe_id, sp.start_node_id, sp.end_node_id, sp.length, sp.systemtype, sp.pipefunction, sp.contentstype,
            sp.material, sp.construction_year, sp.pipeshape, sp.width, sp.height, (gsio.inspection_year - sp.construction_year) as pipe_age, gsio.max_damage_class
        FROM sewer_pipes sp LEFT JOIN (
            SELECT sio.sewer_pipe_id, sio.inspection_year as inspection_year, MAX(sio.damage_class) as max_damage_class
            FROM sewer_inspection_observations sio INNER JOIN (
                SELECT sewer_pipe_id, MAX(inspection_year) as max_inspection_year
                FROM sewer_inspection_observations
                GROUP BY sewer_pipe_id
            ) sio_max_inspection_year
            ON sio.sewer_pipe_id = sio_max_inspection_year.sewer_pipe_id AND sio.inspection_year = sio_max_inspection_year.max_inspection_year
            GROUP BY sio.sewer_pipe_id
        ) gsio
        ON sp.id = gsio.sewer_pipe_id
        {where};"""
        print(f'Sewer pipe with damage class query:\n{query}')
        return self.get_many(query)

    def get_sewers_with_failure_rate(self, min_damage=4):
        query = f"""
        select sp.id, sp.start_node_id, sp.end_node_id, sp.length, sp.systemtype, sp.pipefunction, sp.contentstype, sp.material,
               sp.construction_year, sp.pipeshape, sp.width, sp.height, sio.pipe_age, sio.n_failures, sio.failures_per_meter
        from sewer_pipes sp left join
            (select vsio.sewer_pipe_id, vsio.pipe_age,
            sum(case when damage_class >= 4 then 1 else 0 end)  as n_failures,
            sum(case when damage_class >= 4 then 1 else 0 end) / vsio.pipe_length as failures_per_meter
            from view_sewer_inspection_observations vsio INNER JOIN (
                SELECT sewer_pipe_id, MAX(inspection_year) as max_inspection_year
                FROM sewer_inspection_observations
                GROUP BY sewer_pipe_id
            ) sio_max_inspection_year
            ON vsio.sewer_pipe_id = sio_max_inspection_year.sewer_pipe_id AND vsio.inspection_year = sio_max_inspection_year.max_inspection_year
            where pipe_age >= 0
            group by vsio.sewer_pipe_id order by n_failures) sio
        on sp.id = sio.sewer_pipe_id
        where sp.length is not null and sp.construction_year is not null"""

        print(f'Sewer pipe with failure rate query:\n{query}')
        return self.get_many(query)

    def clear_maintenance_table(self):
        cur = self.get_cursor()
        cur.execute('DELETE FROM maintenance')
        cur.close()
        self.conn.commit()

    def insert_maintenance_actions(self, action_inserts):
        inserts = ',\n'.join(action_inserts)
        query = f"INSERT INTO maintenance (sewer_pipe_id, maintenance_year, maintenance_type, cluster)" \
                f" VALUES \n{inserts};"
        cur = self.get_cursor()
        cur.execute(query)
        cur.close()
        self.conn.commit()

    def get_maintenance_per_year(self):
        query = """SELECT maintenance_year, maintenance_type, count(*)
            FROM maintenance
            WHERE maintenance_type is not 0
            GROUP BY maintenance_year, maintenance_type"""
        return self.get_many(query)

    def get_maintenance_stats(self):
        self.connect()
        stats = {}
        query = """SELECT maintenance_year, maintenance_type, count(*)
                    FROM maintenance
                    WHERE maintenance_type is not 0
                    GROUP BY maintenance_year, maintenance_type"""
        stats['maintenance_per_year'] = self.get_many(query)
        query = """
            SELECT interventions, coalesce(maintenance, 0), coalesce(replacement, 0)
            FROM (SELECT sewer_pipe_id, count(*) as interventions
                    FROM maintenance
                    WHERE maintenance_type != 0
                    GROUP BY sewer_pipe_id) a LEFT JOIN
                (SELECT sewer_pipe_id, count(*) as maintenance
                    FROM maintenance
                    WHERE maintenance_type == 1
                    GROUP BY sewer_pipe_id) b ON a.sewer_pipe_id = b.sewer_pipe_id LEFT JOIN
                (SELECT sewer_pipe_id, count(*) as replacement
                    FROM maintenance
                    WHERE maintenance_type == 2
                    GROUP BY sewer_pipe_id) c ON a.sewer_pipe_id = c.sewer_pipe_id
            ORDER BY interventions;"""
        stats['interventions'] = self.get_many(query)
        query = """
            WITH m AS (
                SELECT maintenance_year, cluster, count(*) n_per_group
                FROM maintenance
                WHERE cluster is not null
                GROUP BY maintenance_year, cluster
            ) {}"""
        stats['avg_pipes_group_year'] = self.get_many(query.format("""
            SELECT maintenance_year, ROUND(AVG(n_per_group), 2) as avg_per_group
            FROM m
            GROUP BY maintenance_year;"""))
        stats['avg_pipes_group_all'] = self.get_many(query.format("""
            SELECT ROUND(AVG(n_per_group), 2) as avg_per_group
            FROM m"""))
        stats['perc_more_than_1_year'] = self.get_many(query.format("""
            SELECT maintenance_year, ROUND(AVG(CASE WHEN n_per_group > 1 THEN 1.0 ELSE 0 END), 2) as ratio_multi_pipe
            FROM m
            GROUP BY maintenance_year;"""))
        stats['perc_more_than_1_all'] = self.get_many(query.format("""
            SELECT ROUND(AVG(CASE WHEN n_per_group > 1 THEN 1.0 ELSE 0 END), 2) as ratio_multi_pipe
            FROM m"""))
        return stats



